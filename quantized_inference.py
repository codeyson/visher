import torch
import yaml
import librosa
import numpy as np
from torch import Tensor
from model import RawNet

NB_SAMP = 64600
SAMPLE_RATE = 24000

BINARY_LABELS = ['fake', 'real']
MULTI_LABELS  = ['gt', 'wavegrad', 'diffwave', 'parallel_wave_gan',
                  'wavernn', 'wavenet', 'melgan']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load model config
with open('model_config_RawNet.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ── Build model, apply quantization structure, THEN load weights ──────────────
# Quantization changes the layer types, so we must recreate the same
# quantized structure before loading the quantized state dict.
model = RawNet(config['model'], device)
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

quantized_model.load_state_dict(
    torch.load('quantized_pruned_model.pth', map_location=device)
)
quantized_model.eval()
print('Quantized model loaded.')


def load_audio(audio_path):
    """Load, resample, pad/trim to NB_SAMP."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) < NB_SAMP:
        y = np.tile(y, (NB_SAMP // len(y)) + 1)
    y = y[:NB_SAMP]
    return Tensor(y).unsqueeze(0)  # (1, NB_SAMP)


def infer(audio_path):
    audio_tensor = load_audio(audio_path).to(device)

    with torch.no_grad():
        output_binary, output_multi = quantized_model(audio_tensor)
        prob_binary = torch.exp(output_binary).squeeze().tolist()
        prob_multi  = torch.exp(output_multi).squeeze().tolist()

    print('\nMulti classification result:')
    print('  ' + ', '.join(f'{l}: {p:.4f}' for l, p in zip(MULTI_LABELS, prob_multi)))

    print('\nBinary classification result:')
    print('  ' + ', '.join(f'{l}: {p:.4f}' for l, p in zip(BINARY_LABELS, prob_binary)))

    pred_binary = BINARY_LABELS[int(torch.argmax(torch.tensor(prob_binary)))]
    pred_multi  = MULTI_LABELS[int(torch.argmax(torch.tensor(prob_multi)))]
    print(f'\n→ Verdict: {pred_binary.upper()}  (most likely source: {pred_multi})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to .wav file')
    args = parser.parse_args()
    infer(args.input_path)