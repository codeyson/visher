"""
benchmark.py
============
Times a single inference pass so we can set a realistic stride
for the live streaming script.

Usage:
    python benchmark.py --model_path ./checkpoints/best_model.pth
    python benchmark.py --model_path ./quantized_pruned_model.pth --quantized
"""

import argparse
import time
import numpy as np
import torch
import yaml
from torch import Tensor
from model import RawNet

NB_SAMP     = 64600
SAMPLE_RATE = 24000
RUNS        = 10   # number of inference passes to average


def load_model(model_path, quantized, device, config):
    model = RawNet(config['model'], device)
    model.eval()
    if quantized:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_benchmark(model, device):
    # Dummy audio — same shape as real input
    dummy = Tensor(np.random.randn(NB_SAMP).astype(np.float32)).unsqueeze(0).to(device)

    # Warmup pass (first pass is always slower due to memory allocation)
    with torch.no_grad():
        model(dummy)

    # Timed passes
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        times.append(time.perf_counter() - start)

    avg   = np.mean(times)
    worst = np.max(times)
    best  = np.min(times)

    print(f'\n── Benchmark Results ({RUNS} runs) ─────────────────────')
    print(f'  Audio window : {NB_SAMP / SAMPLE_RATE:.2f} sec  ({NB_SAMP} samples @ {SAMPLE_RATE}Hz)')
    print(f'  Avg inference: {avg:.3f} sec')
    print(f'  Best         : {best:.3f} sec')
    print(f'  Worst        : {worst:.3f} sec')
    print(f'────────────────────────────────────────────────────')

    # Recommend a stride based on worst-case inference time
    recommended_stride = max(1.0, round(worst + 0.5))
    print(f'\n  Recommended live stream stride: {recommended_stride:.0f} sec')
    if avg < 1.5:
        print('  ✔ Fast enough for near real-time detection.')
    elif avg < 4.0:
        print('  ⚠ Moderate lag expected (~2-4 sec behind speech).')
    else:
        print('  ✘ Slow — consider using the quantized model for live streaming.')
        print('    Run: python benchmark.py --model_path ./quantized_pruned_model.pth --quantized')

    return avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model .pth file')
    parser.add_argument('--quantized',  action='store_true',
                        help='Set this flag if loading a quantized model')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Model : {args.model_path}')
    print(f'Mode  : {"quantized" if args.quantized else "standard"}')

    with open('model_config_RawNet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = load_model(args.model_path, args.quantized, device, config)
    print('Model loaded. Running benchmark...')

    run_benchmark(model, device)