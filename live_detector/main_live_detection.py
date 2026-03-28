"""
main_live_detection.py
Entry point for the real-time AI-voice detection pipeline.

Usage examples
──────────────
# List all audio devices (find your virtual cable)
python main_live_detection.py --list-devices

# Microphone (default)
python main_live_detection.py --model checkpoints/best_model.pth

# Messenger / VoIP via virtual audio cable (VB-Cable)
python main_live_detection.py --model checkpoints/best_model.pth --mic-device "CABLE Output (VB-Audio Virtual Cable)"

# By device index
python main_live_detection.py --model checkpoints/best_model.pth --mic-device 3

# Pre-recorded file (for testing)
python main_live_detection.py --model checkpoints/best_model.pth --source file --path sample.wav

# Raw PCM from stdin (WebRTC / VoIP bridge via ffmpeg)
ffmpeg -i call.sdp -f s16le -ar 16000 -ac 1 - ^
  | python main_live_detection.py --model checkpoints/best_model.pth --source pipe --src-sr 16000

# Show verbose multi-class breakdown
python main_live_detection.py --model checkpoints/best_model.pth --verbose
"""

import argparse
import signal
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio_stream  import get_stream, FileAudioStream, RawPCMStream
from live_inference import LiveDetector, TARGET_SR, TARGET_SAMPLES


# ─────────────────────────────────────────────────────────────
# Device listing
# ─────────────────────────────────────────────────────────────

def list_audio_devices():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"\n{'─'*70}")
        print(f" Available audio INPUT devices")
        print(f"{'─'*70}")
        print(f"  {'IDX':>4}  {'NAME':<45}  {'SR':>8}  {'CH':>3}")
        print(f"{'─'*70}")
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                marker = " ◄ default" if i == sd.default.device[0] else ""
                print(f"  {i:>4}  {d['name'][:45]:<45}  {int(d['default_samplerate']):>8}  {d['max_input_channels']:>3}{marker}")
        print(f"{'─'*70}")
        print(f"\n  For Messenger call detection, look for:")
        print(f"  • 'CABLE Output (VB-Audio Virtual Cable)'")
        print(f"  • 'Stereo Mix' or 'What U Hear' (Realtek)")
        print(f"  • 'VoiceMeeter Output'\n")
    except ImportError:
        print("[error] sounddevice not installed: pip install sounddevice")
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time AI-generated voice detector (RawNet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",        default="checkpoints/best_model.pth",
                   help="Path to RawNet checkpoint (.pth)")
    p.add_argument("--source",       default="mic",
                   choices=["mic", "pyaudio", "file", "pipe"],
                   help="Audio input source")
    p.add_argument("--path",         default=None,
                   help="Audio file path (required when --source=file)")
    p.add_argument("--device",       default=None,
                   help="Torch device: cpu / cuda / cuda:0")
    p.add_argument("--src-sr",       type=int, default=48_000,
                   help="Native sample rate of the capture device (Hz). "
                        "VB-Cable default is 48000; mic is often 16000 or 44100.")
    p.add_argument("--chunk-ms",     type=int, default=100,
                   help="Capture chunk size (ms)")
    p.add_argument("--hop-ms",       type=int, default=1000,
                   help="Sliding-window hop between inferences (ms)")
    p.add_argument("--smoother-k",   type=int, default=5,
                   help="Temporal smoother: majority vote over last K predictions")
    p.add_argument("--verbose",      action="store_true",
                   help="Print multi-class softmax breakdown")
    p.add_argument("--log-json",     default=None,
                   help="Append predictions to a JSON-lines file")
    p.add_argument("--mic-device",   default=None,
                   help="sounddevice device name (partial match OK) or index number")
    p.add_argument("--silence-threshold", type=float, default=0.0001,
                   help="RMS energy below this level is treated as silence and skipped (0=disable)")
    p.add_argument("--list-devices", action="store_true",
                   help="List all audio input devices and exit")
    return p


# ─────────────────────────────────────────────────────────────
# Result printer
# ─────────────────────────────────────────────────────────────

class ResultPrinter:
    def __init__(self, verbose=False, log_path=None):
        self.verbose   = verbose
        self._logfile  = open(log_path, "a") if log_path else None
        self._ai_count = 0
        self._total    = 0

    def __call__(self, result: dict):
        self._total    += 1
        self._ai_count += result["smooth_label"]

        ts   = result["timestamp"]
        lbl  = result["smooth_label_str"]
        conf = result["smooth_confidence"]
        lat  = result["latency_ms"]

        if result["smooth_label"] == 1:
            colour, flag = "\033[91m", "⚠️  AI VOICE DETECTED"
        else:
            colour, flag = "\033[92m", "✅  Human Voice      "

        reset = "\033[0m"
        print(f"[{ts:07.2f}s] {colour}{flag}{reset}  conf={conf:.2f}  lat={lat:.0f}ms")

        if self.verbose:
            top = sorted(result["multi_probs"].items(), key=lambda x: -x[1])
            print("         Multi: " + "  ".join(f"{k}={v:.2f}" for k, v in top[:3]))

        if self._logfile:
            self._logfile.write(json.dumps(result) + "\n")
            self._logfile.flush()

    def summary(self):
        if not self._total:
            return
        pct = 100.0 * self._ai_count / self._total
        print(
            f"\n── Session summary ──────────────────────\n"
            f"  Windows analysed : {self._total}\n"
            f"  AI voice flagged : {self._ai_count}  ({pct:.1f}%)\n"
            f"─────────────────────────────────────────\n"
        )

    def close(self):
        if self._logfile:
            self._logfile.close()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def resolve_device(mic_device):
    """Accept integer index or partial string name."""
    if mic_device is None:
        return None
    try:
        return int(mic_device)
    except (ValueError, TypeError):
        return mic_device   # sounddevice accepts partial name strings


def get_device_samplerate(mic_device) -> int:
    """Query the native sample rate of the chosen device."""
    try:
        import sounddevice as sd
        info = sd.query_devices(mic_device)
        return int(info["default_samplerate"])
    except Exception:
        return 48_000


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    if args.list_devices:
        list_audio_devices()   # exits

    # Auto-detect src-sr from device if user didn't override
    device_id  = resolve_device(args.mic_device)
    src_sr     = args.src_sr
    if device_id is not None and args.src_sr == 48_000:
        detected = get_device_samplerate(device_id)
        if detected != src_sr:
            print(f"[main] Auto-detected device sample rate: {detected} Hz")
            src_sr = detected

    hop_samples = int(TARGET_SR * args.hop_ms / 1000)
    printer     = ResultPrinter(verbose=args.verbose, log_path=args.log_json)

    print(
        f"\n{'─'*60}\n"
        f" RawNet Real-Time AI Voice Detector\n"
        f"{'─'*60}\n"
        f" Model    : {args.model}\n"
        f" Source   : {args.source}"
        + (f"  →  device '{device_id}'" if device_id is not None else "") + "\n"
        f" Src SR   : {src_sr} Hz  →  resample to {TARGET_SR} Hz\n"
        f" Window   : {TARGET_SAMPLES} samples ({TARGET_SAMPLES/TARGET_SR:.2f}s)\n"
        f" Hop      : {hop_samples} samples ({args.hop_ms} ms)\n"
        f" Smoother : last-{args.smoother_k} majority vote\n"
        f"{'─'*60}\n"
        f" Tip: use --list-devices to find your virtual cable index\n"
        f" Press Ctrl+C to stop.\n"
    )

    detector = LiveDetector(
        model_path    =args.model,
        device        =args.device,
        src_sr        =src_sr,
        window_samples=TARGET_SAMPLES,
        hop_samples   =hop_samples,
        smoother_k         =args.smoother_k,
        silence_threshold  =args.silence_threshold,
        on_result          =printer,
    )

    def _shutdown(sig, frame):
        print("\n[main] Shutting down …")
        detector.stop()
        printer.summary()
        printer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── route audio source ────────────────────────────────────────────────

    if args.source == "file":
        if not args.path:
            print("[error] --path is required when --source=file")
            sys.exit(1)
        stream = FileAudioStream(path=args.path, chunk_ms=args.chunk_ms, realtime=True)
        for chunk, sr in stream:
            detector.push(chunk)
        time.sleep(2)

    elif args.source == "pipe":
        stream = RawPCMStream(sample_rate=src_sr, chunk_ms=args.chunk_ms)
        for chunk, sr in stream:
            detector.push(chunk)

    else:
        stream_kwargs = dict(sample_rate=src_sr, chunk_ms=args.chunk_ms)
        if device_id is not None:
            stream_kwargs["device"] = device_id

        stream = get_stream(args.source, **stream_kwargs)
        if args.source == "mic":
            with stream as s:
                for chunk in s:
                    detector.push(chunk)
        else:
            for chunk in stream:
                detector.push(chunk)

    printer.summary()
    printer.close()


if __name__ == "__main__":
    main()