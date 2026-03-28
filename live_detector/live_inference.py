"""
live_inference.py
Preprocessing pipeline + inference engine for real-time AI-voice detection.

Responsibilities
────────────────
1. Resample incoming audio to 24 000 Hz
2. Pad / truncate to exactly 64 600 samples
3. Normalise to float32 in [-1, 1]
4. Run RawNet forward pass on GPU/CPU
5. Return (label, confidence, multi_probs)
"""

import time
import threading
import numpy as np
import torch
import torch.nn.functional as F

from realtime_buffer import SlidingWindowBuffer, TemporalSmoother
from model_loader     import RawNet, load_model, DEFAULT_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SR      = 24_000
TARGET_SAMPLES = 24_000          # ~2.69 s at 24 kHz
BINARY_LABELS  = {0: "Human Voice", 1: "AI Generated Voice"}
MULTI_LABELS   = [
    "gt", "wavegrad", "diffwave",
    "parallel_wave_gen", "wavernn", "wavenet", "melgan",
]


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessor
# ──────────────────────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Converts a variable-length mono float32 array at `src_sr` into a
    (1, 1, 64600) torch.Tensor ready for RawNet inference.
    """

    def __init__(self, src_sr: int = TARGET_SR, target_sr: int = TARGET_SR):
        self.src_sr    = src_sr
        self.target_sr = target_sr

    def resample(self, audio: np.ndarray, src_sr: int | None = None) -> np.ndarray:
        sr = src_sr or self.src_sr
        if sr == self.target_sr:
            return audio
        # Use torchaudio's high-quality resampler when available
        try:
            import torchaudio.functional as TAF
            t   = torch.from_numpy(audio).unsqueeze(0)          # (1, T)
            t   = TAF.resample(t, sr, self.target_sr)
            return t.squeeze(0).numpy()
        except ImportError:
            # Fallback: simple linear interpolation via numpy
            n_out  = int(len(audio) * self.target_sr / sr)
            x_old  = np.linspace(0, 1, len(audio))
            x_new  = np.linspace(0, 1, n_out)
            return np.interp(x_new, x_old, audio).astype(np.float32)

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n == TARGET_SAMPLES:
            return audio
        if n < TARGET_SAMPLES:
            # Repeat-pad (mirrors training preprocessing)
            reps  = TARGET_SAMPLES // n + 1
            audio = np.tile(audio, reps)
            return audio[:TARGET_SAMPLES]
        # Longer than needed: pick the loudest window instead of always
        # taking the first chunk (which may be silent lead-in)
        hop       = TARGET_SAMPLES // 4
        best_rms  = -1.0
        best_start = 0
        for start in range(0, n - TARGET_SAMPLES + 1, hop):
            chunk = audio[start : start + TARGET_SAMPLES]
            rms   = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > best_rms:
                best_rms   = rms
                best_start = start
        return audio[best_start : best_start + TARGET_SAMPLES]

    def normalise(self, audio: np.ndarray) -> np.ndarray:
        # Use 99th-percentile peak instead of max to be robust to
        # single-sample spikes from VB-Cable/Windows audio processing
        peak = np.abs(audio).max()   # ← match training
        if peak > 1e-6:
            audio = audio / peak
        return audio.astype(np.float32)

    def process(self, audio: np.ndarray, src_sr: int | None = None) -> torch.Tensor:
        audio = self.resample(audio, src_sr)
        audio = self.pad_or_truncate(audio)
        audio = self.normalise(audio)
        t = torch.from_numpy(audio).unsqueeze(0)  # (1, T) — model reshapes internally
        return t


# ──────────────────────────────────────────────────────────────────────────────
# Single-chunk inference
# ──────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Wraps the loaded RawNet model and exposes a predict() method.
    """

    def __init__(
        self,
        model    : RawNet,
        device   : str,
        smoother : TemporalSmoother | None = None,
    ):
        self.model    = model
        self.device   = device
        self.smoother = smoother or TemporalSmoother(window_size=5, mode="majority")
        self._preproc = Preprocessor()

    @torch.inference_mode()
    def predict(
        self,
        window   : np.ndarray,
        src_sr   : int | None = None,
        timestamp: float = 0.0,
    ) -> dict:
        """
        Run inference on one audio window.

        Parameters
        ----------
        window    : mono float32 numpy array (raw samples).
        src_sr    : source sample rate; None means already at 24 kHz.
        timestamp : wall-clock offset in seconds (for display).

        Returns
        -------
        dict with keys:
            timestamp, raw_label, raw_confidence,
            smooth_label, smooth_confidence,
            multi_probs, latency_ms
        """
        t0 = time.perf_counter()

        tensor = self._preproc.process(window, src_sr).to(self.device)

        out_bin, out_multi = self.model(tensor)

        # Binary
        probs_bin   = torch.exp(out_bin)[0]  # model uses logsoftmax
        raw_label   = int(torch.argmax(probs_bin).item())
        raw_conf    = float(probs_bin[raw_label].item())

        # Multi-class
        probs_multi = torch.exp(out_multi)[0].cpu().numpy()

        # Temporal smoothing
        smooth_label, smooth_conf = self.smoother.update(raw_label, raw_conf)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "timestamp"        : timestamp,
            "raw_label"        : raw_label,
            "raw_label_str"    : BINARY_LABELS[raw_label],
            "raw_confidence"   : raw_conf,
            "smooth_label"     : smooth_label,
            "smooth_label_str" : BINARY_LABELS[smooth_label],
            "smooth_confidence": smooth_conf,
            "multi_probs"      : {k: float(v) for k, v in zip(MULTI_LABELS, probs_multi)},
            "latency_ms"       : latency_ms,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Streaming detector (ties buffer + engine together)
# ──────────────────────────────────────────────────────────────────────────────

class LiveDetector:
    """
    High-level detector that:
    • accepts raw audio pushes from the stream thread
    • runs inference in a background thread
    • calls a result callback with each prediction

    Parameters
    ----------
    model_path    : str          – path to best_model.pth
    device        : str | None   – 'cuda', 'cpu', or None (auto)
    src_sr        : int          – sample rate of incoming audio
    window_samples: int          – model input length (samples at 24 kHz)
    hop_samples   : int          – hop size (samples at 24 kHz)
    smoother_k    : int          – temporal smoother window
    on_result     : callable     – called with the result dict for each window
    """

    def __init__(
        self,
        model_path    : str,
        device        : str | None  = None,
        src_sr        : int         = TARGET_SR,
        window_samples: int         = TARGET_SAMPLES,
        hop_samples   : int         = 24_000,
        smoother_k    : int         = 5,
        on_result                   = None,
        silence_threshold   : float = 0.001,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device   = device
        self.src_sr   = src_sr

        # Components
        model         = load_model(model_path, config=None, device=device)
        smoother      = TemporalSmoother(window_size=smoother_k)
        self.engine   = InferenceEngine(model, device, smoother)

        self.buffer   = SlidingWindowBuffer(
            window_samples=window_samples,
            hop_samples   =hop_samples,
        )

        self.silence_threshold = silence_threshold
        self.on_result = on_result or self._default_print
        self._start_time = time.time()

        # Inference runs in a daemon thread so it never blocks the audio thread
        self._infer_q : queue.Queue[np.ndarray] = __import__("queue").Queue(maxsize=8)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    # ── public ──────────────────────────────────────────────────────────────

    def push(self, samples: np.ndarray) -> None:
        """Feed new raw audio samples (at src_sr)."""
        # If src_sr != TARGET_SR pre-resample here so the buffer holds 24 kHz samples
        if self.src_sr != TARGET_SR:
            samples = self.engine._preproc.resample(samples, self.src_sr)
        self.buffer.push(samples)

        # Drain all ready windows into the inference queue (non-blocking)
        for window in self.buffer.drain_windows():
            try:
                self._infer_q.put_nowait(window)
            except __import__("queue").Full:
                pass   # drop oldest if backpressure too high

    def stop(self) -> None:
        self._stop.set()

    # ── private ─────────────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        import queue as _queue
        segment_votes = []   # accumulate votes for current speech segment

        while not self._stop.is_set():
            try:
                window = self._infer_q.get(timeout=0.2)
                rms = float(np.sqrt(np.mean(window ** 2)))

                if rms < self.silence_threshold:
                    # End of speech segment — emit verdict if we have votes
                    if segment_votes:
                        ai_ratio = sum(segment_votes) / len(segment_votes)
                        verdict = 1 if ai_ratio >= 0.34 else 0
                        label_str = "AI Generated Voice" if verdict == 1 else "Human Voice"
                        flag = "⚠️ " if verdict == 1 else "✅"
                        print(f"\n{'='*50}")
                        print(f"  SEGMENT VERDICT: {flag} {label_str}")
                        print(f"  AI windows: {sum(segment_votes)}/{len(segment_votes)} ({ai_ratio*100:.0f}%)")
                        print(f"{'='*50}\n")
                        segment_votes = []
                    self.engine.smoother.reset()
                    print(f"[{time.time()-self._start_time:07.2f}s] 🔇  Silence (rms={rms:.5f}) — skipping")
                    continue

                timestamp = time.time() - self._start_time
                result = self.engine.predict(window, src_sr=None, timestamp=timestamp)
                result["window_rms"] = rms

                # Accumulate raw (not smoothed) per-window vote
                segment_votes.append(result["raw_label"])

                self.on_result(result)
            except _queue.Empty:
                continue
    @staticmethod
    def _default_print(result: dict) -> None:
        ts   = result["timestamp"]
        lbl  = result["smooth_label_str"]
        conf = result["smooth_confidence"]
        lat  = result["latency_ms"]
        flag = "⚠️ " if result["smooth_label"] == 1 else "✅ "
        print(f"[{ts:07.2f}s] {flag}{lbl:25s} (confidence {conf:.2f}, latency {lat:.0f} ms)")