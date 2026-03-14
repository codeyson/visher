"""
live_stream.py
==============
Real-time voice phishing detection using a sliding window over live mic input.

Usage:
    python live_stream.py --model_path ./checkpoints/best_model.pth

Controls:
    Ctrl+C  →  stop
"""

import argparse
import time
import threading
import collections
import numpy as np
import torch
import yaml
import sounddevice as sd
from torch import Tensor
from model import RawNet

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE       = 24000
NB_SAMP           = 64600
STRIDE_SEC        = 1.0
BLOCK_SIZE        = 2400
SILENCE_THRESHOLD = 0.01

BINARY_LABELS = ['real', 'fake']
MULTI_LABELS  = ['gt', 'wavegrad', 'diffwave', 'parallel_wave_gan',
                  'wavernn', 'wavenet', 'melgan']

CONFIDENCE_THRESHOLD = 0.65

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path, device, config):
    model = RawNet(config['model'], device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# ── Silence detection ─────────────────────────────────────────────────────────

def is_silent(audio_window, threshold=SILENCE_THRESHOLD):
    rms = np.sqrt(np.mean(audio_window ** 2))
    return rms < threshold


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, audio_window, device, real_bias=0.0):
    """
    Run model on a NB_SAMP numpy array.
    real_bias: add this value to the real class probability before deciding.
               Increase (e.g. 0.1-0.3) if model is too aggressive about fake.
    """
    x = Tensor(audio_window.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out_binary, out_multi = model(x)
        prob_binary = torch.exp(out_binary).squeeze().tolist()
        prob_multi  = torch.exp(out_multi).squeeze().tolist()

    # Apply real bias to compensate for class imbalance in training
    prob_real = min(1.0, prob_binary[0] + real_bias)
    prob_fake = 1.0 - prob_real
    prob_binary = [prob_real, prob_fake]

    pred_idx   = int(np.argmax(prob_binary))
    verdict    = BINARY_LABELS[pred_idx]
    confidence = prob_binary[pred_idx]
    source     = MULTI_LABELS[int(np.argmax(prob_multi))]

    return verdict, confidence, source, prob_binary, prob_multi


# ── Display ───────────────────────────────────────────────────────────────────

def print_verdict(verdict, confidence, source, prob_binary, elapsed, threshold):
    bar_len = 30
    filled  = int(bar_len * confidence)
    bar     = '█' * filled + '░' * (bar_len - filled)

    if confidence < threshold:
        label = '  UNCERTAIN '
    elif verdict == 'real':
        label = '  ✔  REAL   '
    else:
        label = '  ✘  FAKE   '

    real_pct = prob_binary[0] * 100
    fake_pct = prob_binary[1] * 100

    print(f'\r[{bar}] {confidence*100:5.1f}%  |{label}|  '
          f'real: {real_pct:5.1f}%  fake: {fake_pct:5.1f}%  '
          f'source: {source:<20s}  ({elapsed:.2f}s)',
          end='', flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='VoiceGuard live stream detector')
    parser.add_argument('--model_path',   type=str,   default='./checkpoints/best_model.pth')
    parser.add_argument('--stride',       type=float, default=STRIDE_SEC)
    parser.add_argument('--threshold',    type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for UNCERTAIN verdict (default: 0.65)')
    parser.add_argument('--silence',      type=float, default=SILENCE_THRESHOLD,
                        help='RMS below this = silence, skip inference (default: 0.01)')
    parser.add_argument('--real_bias',    type=float, default=0.15,
                        help='Boost real class probability to counter fake bias (default: 0.15)')
    parser.add_argument('--device_id',    type=int,   default=None)
    parser.add_argument('--list_devices', action='store_true',
                        help='Print available audio input devices and exit')
    args = parser.parse_args()

    if args.list_devices:
        print('\nAvailable audio input devices:')
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f'  [{i}] {dev["name"]}  '
                      f'(channels: {dev["max_input_channels"]}, '
                      f'default SR: {int(dev["default_samplerate"])}Hz)')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device     : {device}')
    print(f'Model      : {args.model_path}')
    print(f'Window     : {NB_SAMP / SAMPLE_RATE:.2f} sec  ({NB_SAMP} samples)')
    print(f'Stride     : {args.stride} sec')
    print(f'Threshold  : {args.threshold * 100:.0f}% confidence')
    print(f'Silence RMS: {args.silence}')
    print(f'Real bias  : +{args.real_bias:.2f} (set 0.0 to disable)')
    print()

    with open('model_config_RawNet.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = load_model(args.model_path, device, config)
    print('Model loaded. Listening...')
    print('Press Ctrl+C to stop.\n')
    print('-' * 90)

    buffer = collections.deque(maxlen=NB_SAMP)
    buffer.extend(np.zeros(NB_SAMP, dtype=np.float32))

    new_samples_lock  = threading.Lock()
    new_samples_count = [0]
    stride_samples    = int(SAMPLE_RATE * args.stride)

    def audio_callback(indata, frames, time_info, status):
        audio = indata[:, 0].astype(np.float32)
        buffer.extend(audio)
        with new_samples_lock:
            new_samples_count[0] += len(audio)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            dtype='float32',
            device=args.device_id,
            callback=audio_callback,
        ):
            while True:
                with new_samples_lock:
                    ready = new_samples_count[0] >= stride_samples

                if not ready:
                    time.sleep(0.05)
                    continue

                with new_samples_lock:
                    new_samples_count[0] = 0

                window = np.array(buffer, dtype=np.float32)

                if is_silent(window, threshold=args.silence):
                    print('\r[..............................] '
                          'SILENT — waiting for speech...          '
                          '                                        ',
                          end='', flush=True)
                    continue

                t0 = time.perf_counter()
                verdict, confidence, source, prob_binary, prob_multi = \
                    run_inference(model, window, device, real_bias=args.real_bias)
                elapsed = time.perf_counter() - t0

                print_verdict(verdict, confidence, source,
                              prob_binary, elapsed, args.threshold)

    except KeyboardInterrupt:
        print('\n\nStopped.')


if __name__ == '__main__':
    main()