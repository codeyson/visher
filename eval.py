"""
How to run:
    Evaluate Real voice: python eval.py --input_path ./LibriSeVoc/gt/250_142286_000031_000006.wav --model_path ./checkpoints/best_model.pth
    Evaluate Fake voice: python eval.py --input_path ./LibriSeVoc/wavernn/696_92939_000008_000001_gen.wav --model_path ./checkpoints/best_model.pth
"""

import argparse
import numpy as np
import torch
from torch import Tensor
import yaml
import librosa
from model import RawNet

SAMPLE_RATE = 24000
NB_SAMP     = 64600   # must match model_config_RawNet.yaml nb_samp

BINARY_LABELS = ['real', 'fake']
MULTI_LABELS  = ['gt', 'wavegrad', 'diffwave', 'parallel_wave_gen',
                  'wavernn', 'wavenet', 'melgan']


def pad(x, max_len=NB_SAMP):
    """Repeat-pad a 1-D numpy array to max_len."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = (max_len // x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def load_segments(audio_path, max_len=NB_SAMP):
    """
    Load an audio file and return a list of fixed-length Tensor segments.
    Files shorter than max_len are padded to exactly one segment.
    Longer files are chunked into non-overlapping segments of max_len.
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    if len(y) <= max_len:
        return [Tensor(pad(y, max_len))]

    segments = []
    num_chunks = len(y) // max_len
    for i in range(num_chunks):
        chunk = y[i * max_len : (i + 1) * max_len]
        segments.append(Tensor(pad(chunk, max_len)))

    return segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',  type=str, required=True,
                        help='Path to input .wav file')
    parser.add_argument('--model_path',  type=str, required=True,
                        help='Path to saved model .pth file')
    parser.add_argument('--config_path', type=str, default='model_config_RawNet.yaml',
                        help='Path to model config YAML')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load model
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = RawNet(config['model'], device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f'Model loaded: {args.model_path}')

    # Load audio segments
    segments = load_segments(args.input_path)
    print(f'Audio split into {len(segments)} segment(s) of {NB_SAMP} samples each.')

    out_list_binary = []
    out_list_multi  = []

    with torch.no_grad():
        for segment in segments:
            x = segment.to(device=device, dtype=torch.float).unsqueeze(0)  # (1, NB_SAMP)
            out_binary, out_multi = model(x)

            # Model outputs LogSoftmax → use exp() to get probabilities (NOT softmax again)
            prob_binary = torch.exp(out_binary).squeeze().cpu().tolist()
            prob_multi  = torch.exp(out_multi).squeeze().cpu().tolist()

            out_list_binary.append(prob_binary)
            out_list_multi.append(prob_multi)

    # Average probabilities across all segments
    result_binary = np.mean(out_list_binary, axis=0).tolist()
    result_multi  = np.mean(out_list_multi,  axis=0).tolist()

    print('\nMulti classification result:')
    multi_str = ', '.join(
        f'{label}: {prob:.4f}' for label, prob in zip(MULTI_LABELS, result_multi)
    )
    print(f'  {multi_str}')

    print('\nBinary classification result:')
    binary_str = ', '.join(
        f'{label}: {prob:.4f}' for label, prob in zip(BINARY_LABELS, result_binary)
    )
    print(f'  {binary_str}')

    pred_binary = BINARY_LABELS[int(np.argmax(result_binary))]
    pred_multi  = MULTI_LABELS[int(np.argmax(result_multi))]
    print(f'\n→ Verdict: {pred_binary.upper()}  (most likely source: {pred_multi})')