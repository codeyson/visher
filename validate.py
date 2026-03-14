"""
validate.py
===========
Runs the trained model on the held-out test split and prints a full
confusion matrix + accuracy breakdown so you can verify it's working.

Usage:
    python validate.py --data_path ./LibriSeVoc --model_path ./checkpoints/best_model.pth
"""

import argparse
import os
import random
import numpy as np
import torch
import yaml
import scipy.io.wavfile as wavfile
import librosa
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from model import RawNet

SAMPLE_RATE = 24000
NB_SAMP     = 64600

BINARY_LABELS = ['real', 'fake']
MULTI_LABELS  = ['gt', 'wavegrad', 'diffwave', 'parallel_wave_gan',
                  'wavernn', 'wavenet', 'melgan']

FOLDER_TO_LABEL = {
    'gt':                 0,
    'wavegrad':           1,
    'diffwave':           2,
    'parallel_wave_gen':  3,
    'parallel_wave_gan':  3,
    'wavernn':            4,
    'wavenet':            5,
    'melgan':             6,
}


# ── Dataset (copied from main.py) ─────────────────────────────────────────────

class Dataset_LibriSeVoc(Dataset):
    def __init__(self, file_list, label_binary, label_multi):
        self.file_list    = file_list
        self.label_binary = label_binary
        self.label_multi  = label_multi
        self.cut          = NB_SAMP

    def __len__(self):
        return len(self.file_list)

    def load_waveform(self, audio_path):
        if os.path.getsize(audio_path) < 44:
            return None
        y, sr = None, None
        try:
            sr, data = wavfile.read(audio_path)
            if data is None or len(data) == 0:
                return None
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)
            if data.ndim > 1:
                data = data.mean(axis=1)
            y = data
        except Exception:
            pass
        if y is None:
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            except Exception:
                return None
        if y is None or len(y) == 0:
            return None
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        if len(y) < self.cut:
            y = np.tile(y, (self.cut // len(y)) + 1)
        y = y[:self.cut]
        return Tensor(y)

    def __getitem__(self, index):
        return (self.load_waveform(self.file_list[index]),
                self.label_binary[index],
                self.label_multi[index])


def collate_skip_none(batch):
    batch = [(w, b, m) for w, b, m in batch if w is not None]
    if len(batch) == 0:
        return None, None, None
    return (torch.stack([x[0] for x in batch]),
            torch.tensor([x[1] for x in batch], dtype=torch.long),
            torch.tensor([x[2] for x in batch], dtype=torch.long))


def build_file_lists(dataset_path, train_ratio=0.7, dev_ratio=0.15, seed=42):
    all_paths, all_binary, all_multi = [], [], []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        folder_key = folder_name.lower()
        if folder_key not in FOLDER_TO_LABEL:
            continue
        multi_label  = FOLDER_TO_LABEL[folder_key]
        binary_label = 0 if folder_key == 'gt' else 1
        wav_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.wav')
            and os.path.getsize(os.path.join(folder_path, f)) >= 44
        ]
        all_paths   += wav_files
        all_binary  += [binary_label] * len(wav_files)
        all_multi   += [multi_label]  * len(wav_files)

    rng = random.Random(seed)
    combined = list(zip(all_paths, all_binary, all_multi))
    rng.shuffle(combined)
    all_paths, all_binary, all_multi = zip(*combined)

    n       = len(all_paths)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)

    return {
        'train': (all_paths[:n_train],          all_binary[:n_train],          all_multi[:n_train]),
        'dev':   (all_paths[n_train:n_train+n_dev], all_binary[n_train:n_train+n_dev], all_multi[n_train:n_train+n_dev]),
        'test':  (all_paths[n_train+n_dev:],    all_binary[n_train+n_dev:],    all_multi[n_train+n_dev:]),
    }


# ── Confusion matrix printer ──────────────────────────────────────────────────

def print_confusion_matrix(matrix, labels):
    col_w = 14
    print(f"\n{'':>{col_w}}", end='')
    for l in labels:
        print(f'{l:>{col_w}}', end='')
    print(f"{'Total':>{col_w}}")
    for i, row_label in enumerate(labels):
        print(f'{row_label:>{col_w}}', end='')
        for j in range(len(labels)):
            cell = matrix[i][j]
            marker = ' ←' if i == j else ''
            print(f'{str(cell) + marker:>{col_w}}', end='')
        print(f'{sum(matrix[i]):>{col_w}}')
    print()
    for i, label in enumerate(labels):
        total   = sum(matrix[i])
        correct = matrix[i][i]
        pct     = (correct / total * 100) if total > 0 else 0
        print(f'  {label:<22s}: {correct:>4}/{total:<4}  ({pct:.1f}%)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   type=str,   required=True)
    parser.add_argument('--model_path',  type=str,   required=True)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--dev_ratio',   type=float, default=0.15)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    print('\nScanning dataset...')
    splits      = build_file_lists(args.data_path, args.train_ratio, args.dev_ratio)
    test_set    = Dataset_LibriSeVoc(*splits['test'])
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=collate_skip_none)
    print(f'Test samples: {len(test_set)}')

    with open('model_config_RawNet.yaml', 'r') as f:
        config = yaml.safe_load(f)
    model = RawNet(config['model'], device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f'Model loaded: {args.model_path}\n')

    cm_bin   = [[0] * len(BINARY_LABELS) for _ in BINARY_LABELS]
    cm_multi = [[0] * len(MULTI_LABELS)  for _ in MULTI_LABELS]
    correct_bin, correct_multi, total = 0, 0, 0

    with torch.no_grad():
        for waveforms, labels_bin, labels_multi in test_loader:
            if waveforms is None:
                continue
            waveforms    = waveforms.to(device)
            labels_bin   = labels_bin.to(device)
            labels_multi = labels_multi.to(device)

            out_binary, out_multi = model(waveforms)
            _, preds_bin   = torch.max(out_binary, 1)
            _, preds_multi = torch.max(out_multi,  1)

            for true, pred in zip(labels_bin.cpu(), preds_bin.cpu()):
                cm_bin[true.item()][pred.item()] += 1
            for true, pred in zip(labels_multi.cpu(), preds_multi.cpu()):
                cm_multi[true.item()][pred.item()] += 1

            correct_bin   += (preds_bin   == labels_bin).sum().item()
            correct_multi += (preds_multi == labels_multi).sum().item()
            total         += labels_bin.size(0)

    print('=' * 65)
    print(f'  BINARY CLASSIFICATION  (real vs fake)')
    print(f'  Overall accuracy: {correct_bin/total*100:.2f}%  ({correct_bin}/{total})')
    print('=' * 65)
    print_confusion_matrix(cm_bin, BINARY_LABELS)

    print('\n' + '=' * 65)
    print(f'  MULTI-CLASS CLASSIFICATION  (vocoder source)')
    print(f'  Overall accuracy: {correct_multi/total*100:.2f}%  ({correct_multi}/{total})')
    print('=' * 65)
    print_confusion_matrix(cm_multi, MULTI_LABELS)

    binary_acc = correct_bin / total * 100
    print('\n── Sanity Check ──────────────────────────────────────────')
    if binary_acc >= 90:
        print(f'  ✔ Model is working well  ({binary_acc:.1f}% binary accuracy)')
        print('    Live stream results can be trusted.')
    elif binary_acc >= 70:
        print(f'  ⚠ Model is working but could be better  ({binary_acc:.1f}%)')
        print('    Consider training more epochs.')
    else:
        print(f'  ✘ Model accuracy is low  ({binary_acc:.1f}%)')
        print('    Train longer before using live stream.')
    print('──────────────────────────────────────────────────────────')


if __name__ == '__main__':
    main()