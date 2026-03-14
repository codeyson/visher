"""
audio_stream.py
Audio capture backends for microphone, file, and raw PCM / WebRTC piped input.
All backends expose the same iterator interface:

    for chunk in backend:
        buffer.push(chunk)

Each chunk is a numpy float32 array of shape (N,) at the *original* device
sample rate. Resampling to 24 kHz is handled by the preprocessor in
live_inference.py so that the backend stays thin and testable.
"""

import queue
import threading
import numpy as np
from pathlib import Path
from typing import Iterator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    """Convert any int/float multi-channel array to mono float32 in [-1, 1]."""
    # Normalise integer types
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    else:
        data = data.astype(np.float32)

    # Mix down to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Microphone stream  (sounddevice)
# ──────────────────────────────────────────────────────────────────────────────

class MicrophoneStream:
    """
    Capture audio from the default (or specified) microphone via sounddevice.

    Parameters
    ----------
    device      : int | str | None  –  sounddevice device index/name or None for default.
    sample_rate : int               –  device capture rate (will be resampled later).
    chunk_ms    : int               –  length of each chunk in milliseconds.
    channels    : int               –  number of input channels (1 or 2).
    """

    def __init__(
        self,
        device     : int | str | None = None,
        sample_rate: int = 16_000,
        chunk_ms   : int = 100,
        channels   : int = 1,
    ):
        self.device      = device
        self.sample_rate = sample_rate
        self.chunk_ms    = chunk_ms
        self.channels    = channels
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self._stop   = threading.Event()

    # ── sounddevice callback (called from audio thread) ───────────────────
    def _callback(self, indata, frames, time_info, status):
        if status:
            import sys; print(f"[audio_stream] {status}", file=sys.stderr)
        self._q.put(indata.copy())

    # ── context manager ───────────────────────────────────────────────────
    def __enter__(self):
        import sounddevice as sd
        blocksize = int(self.sample_rate * self.chunk_ms / 1000)
        self._stream = sd.InputStream(
            device    = self.device,
            samplerate= self.sample_rate,
            channels  = self.channels,
            dtype     = "float32",
            blocksize = blocksize,
            callback  = self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._stop.set()

    def __iter__(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.5)
                yield _to_mono_float32(chunk)
            except queue.Empty:
                continue

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────────────────────────────────────
# PyAudio microphone stream (fallback)
# ──────────────────────────────────────────────────────────────────────────────

class PyAudioStream:
    """Microphone capture via PyAudio (fallback when sounddevice is unavailable)."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_ms   : int = 100,
        channels   : int = 1,
    ):
        self.sample_rate = sample_rate
        self.chunk_ms    = chunk_ms
        self.channels    = channels
        self._stop       = threading.Event()

    def __iter__(self) -> Iterator[np.ndarray]:
        import pyaudio
        pa        = pyaudio.PyAudio()
        blocksize = int(self.sample_rate * self.chunk_ms / 1000)
        stream    = pa.open(
            format           = pyaudio.paFloat32,
            channels         = self.channels,
            rate             = self.sample_rate,
            input            = True,
            frames_per_buffer= blocksize,
        )
        try:
            while not self._stop.is_set():
                raw   = stream.read(blocksize, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32)
                yield _to_mono_float32(chunk)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────────────────────────────────────
# File / pre-recorded stream  (for testing & replay)
# ──────────────────────────────────────────────────────────────────────────────

class FileAudioStream:
    """
    Read a WAV / MP3 / FLAC / OGG file and emit chunks, optionally at real-time speed.

    Requires torchaudio or soundfile.
    """

    def __init__(
        self,
        path           : str | Path,
        chunk_ms       : int  = 100,
        realtime       : bool = True,
    ):
        self.path     = Path(path)
        self.chunk_ms = chunk_ms
        self.realtime = realtime
        self._stop    = threading.Event()

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        """Yields (mono_float32_chunk, sample_rate) tuples."""
        import time
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(self.path))
            audio = waveform.mean(dim=0).numpy().astype(np.float32)
        except Exception:
            import soundfile as sf
            audio, sr = sf.read(str(self.path), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)

        blocksize  = int(sr * self.chunk_ms / 1000)
        chunk_dur  = self.chunk_ms / 1000.0

        for start in range(0, len(audio), blocksize):
            if self._stop.is_set():
                break
            chunk = audio[start : start + blocksize]
            yield chunk, sr        # caller must resample to 24 kHz
            if self.realtime:
                time.sleep(chunk_dur)

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────────────────────────────────────
# Raw PCM / pipe stream  (WebRTC, VoIP, RTP)
# ──────────────────────────────────────────────────────────────────────────────

class RawPCMStream:
    """
    Read raw signed-16-bit PCM from stdin or a named pipe.
    Useful for integration with WebRTC, Asterisk, or RTP bridges that pipe
    raw audio over stdout.

    Usage:
        ffmpeg -i sip_stream.sdp -f s16le -ar 16000 -ac 1 - | python main_live_detection.py --source pipe

    Parameters
    ----------
    sample_rate  : int   –  PCM sample rate (Hz).
    chunk_ms     : int   –  chunk duration in ms.
    source       : str   –  'stdin' or a path to a named pipe / FIFO.
    """

    def __init__(
        self,
        sample_rate: int  = 16_000,
        chunk_ms   : int  = 100,
        source     : str  = "stdin",
    ):
        self.sample_rate = sample_rate
        self.chunk_ms    = chunk_ms
        self.source      = source
        self._stop       = threading.Event()

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        import sys
        blocksize   = int(self.sample_rate * self.chunk_ms / 1000)
        byte_count  = blocksize * 2     # int16 → 2 bytes per sample

        fh = sys.stdin.buffer if self.source == "stdin" else open(self.source, "rb")
        try:
            while not self._stop.is_set():
                raw = fh.read(byte_count)
                if not raw:
                    break
                chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                yield chunk, self.sample_rate
        finally:
            if self.source != "stdin":
                fh.close()

    def stop(self):
        self._stop.set()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_stream(source: str = "mic", **kwargs):
    """
    Factory returning an appropriate stream object.

    source : 'mic'   → MicrophoneStream (sounddevice)
             'pyaudio'→ PyAudioStream
             'file'  → FileAudioStream  (requires kwarg: path=...)
             'pipe'  → RawPCMStream     (stdin or named pipe)
    """
    if source == "mic":
        return MicrophoneStream(**kwargs)
    elif source == "pyaudio":
        return PyAudioStream(**kwargs)
    elif source == "file":
        return FileAudioStream(**kwargs)
    elif source == "pipe":
        return RawPCMStream(**kwargs)
    else:
        raise ValueError(f"Unknown source: '{source}'")