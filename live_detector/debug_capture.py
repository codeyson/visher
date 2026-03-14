"""
debug_capture.py
Captures ~5 seconds from VB-Cable, saves it as a WAV, then runs the model
on it directly — so you can compare live capture vs file-based prediction.

Usage:
    python live_detector/debug_capture.py --model checkpoints/best_model.pth --mic-device 17

Play your TTS audio WHILE this script is running. It will:
1. Record 5s from the device
2. Save it to debug_capture.wav
3. Run the model on it and print probabilities
4. Also run test_model.py on the saved file so you can compare
"""

import sys, argparse, time, queue, threading
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import sounddevice as sd
import soundfile as sf
from model_loader import load_model

SAMPLE_RATE  = 24_000
TARGET_SAMP  = 64_600
RECORD_SECS  = 6


def resample(audio, src_sr, dst_sr=SAMPLE_RATE):
    if src_sr == dst_sr:
        return audio
    try:
        import torchaudio.functional as TAF
        t = torch.from_numpy(audio).unsqueeze(0)
        t = TAF.resample(t, src_sr, dst_sr)
        return t.squeeze(0).numpy()
    except Exception:
        n = int(len(audio) * dst_sr / src_sr)
        return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(audio)), audio).astype('float32')


def record(device, src_sr, seconds):
    print(f"\n🎙  Recording {seconds}s from device '{device}' at {src_sr} Hz ...")
    print("    ▶  Play your TTS audio NOW\n")
    chunks = []
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata.copy())

    with sd.InputStream(device=device, samplerate=src_sr,
                        channels=1, dtype='float32',
                        blocksize=int(src_sr * 0.1),
                        callback=cb):
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                chunks.append(q.get(timeout=0.2))
            except queue.Empty:
                pass

    audio = np.concatenate(chunks).flatten().astype('float32')
    print(f"    Recorded {len(audio)/src_sr:.2f}s  |  "
          f"RMS={np.sqrt(np.mean(audio**2)):.5f}  |  Peak={np.abs(audio).max():.5f}")
    return audio, src_sr


def preprocess(wav, src_sr):
    # Step 1: resample
    wav = resample(wav, src_sr)
    print(f"\n── After resample to {SAMPLE_RATE} Hz ──────────────────")
    print(f"   Samples: {len(wav)}  RMS={np.sqrt(np.mean(wav**2)):.5f}  Peak={np.abs(wav).max():.5f}")

    # Step 2: pad/truncate — pick loudest window to avoid silent lead-in
    if len(wav) < TARGET_SAMP:
        wav = np.tile(wav, TARGET_SAMP // len(wav) + 1)
        wav = wav[:TARGET_SAMP]
    else:
        hop, best_rms, best_start = TARGET_SAMP // 4, -1.0, 0
        for s in range(0, len(wav) - TARGET_SAMP + 1, hop):
            rms = float(np.sqrt(np.mean(wav[s:s+TARGET_SAMP]**2)))
            if rms > best_rms:
                best_rms, best_start = rms, s
        wav = wav[best_start : best_start + TARGET_SAMP]

    # Step 3: normalise
    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak
    print(f"── After normalise ─────────────────────────────────")
    print(f"   Samples: {len(wav)}  RMS={np.sqrt(np.mean(wav**2)):.5f}  Peak={np.abs(wav).max():.5f}")
    return wav.astype('float32')


def predict(model, wav, device_str):
    x = torch.from_numpy(wav).unsqueeze(0).to(device_str)
    with torch.inference_mode():
        out_bin, out_multi = model(x)
    probs = torch.exp(out_bin)[0].cpu()
    label = int(torch.argmax(probs))
    labels = ['Human Voice', 'AI Generated']
    print(f"\n── Model prediction ────────────────────────────────")
    print(f"   Human prob : {probs[0]:.4f}")
    print(f"   AI    prob : {probs[1]:.4f}")
    print(f"   Decision   : {'✅ Human Voice' if label==0 else '⚠️  AI Voice'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      required=True)
    p.add_argument('--mic-device', default=17, help='sounddevice device index')
    p.add_argument('--src-sr',     type=int, default=48_000)
    p.add_argument('--output',     default='debug_capture.wav')
    p.add_argument('--seconds',    type=int, default=RECORD_SECS)
    args = p.parse_args()

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model, device=torch_device)

    # Try to parse device as int
    try:
        dev = int(args.mic_device)
    except (ValueError, TypeError):
        dev = args.mic_device

    # Auto-detect src_sr from device
    try:
        info = sd.query_devices(dev)
        src_sr = int(info['default_samplerate'])
        print(f"[debug] Device native SR: {src_sr} Hz")
    except Exception:
        src_sr = args.src_sr

    raw, sr = record(dev, src_sr, args.seconds)

    # Save raw capture for inspection
    sf.write(args.output, raw, sr)
    print(f"\n💾  Raw capture saved → {args.output}  (open in Audacity to inspect)")

    # Also save resampled version
    resampled = resample(raw, sr)
    sf.write('debug_capture_24k.wav', resampled, SAMPLE_RATE)
    print(f"💾  Resampled (24kHz) saved → debug_capture_24k.wav")

    # Run model
    wav = preprocess(raw, sr)
    predict(model, wav, torch_device)

    print(f"\n📋  Now run this to compare with direct file load:")
    print(f"    python live_detector/test_model.py --model {args.model} --file debug_capture_24k.wav")
    print()


if __name__ == '__main__':
    main()