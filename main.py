"""
Train_visher : python main.py --data_path ./LibriSeVoc --batch_size 4 --num_epochs 2 --model_save_path ./checkpoints
Train_voiceguard: python main.py --data_path ./LibriSeVoc --batch_size 64 --num_epochs 50 --lr 0.0001 --model_save_path ./checkpoints
"""

import argparse
import os
import random
import numpy as np
import yaml
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from model import RawNet
import librosa
import scipy.io.wavfile as wavfile
from tqdm import tqdm

SAMPLE_RATE = 24000
# RawNet expects exactly nb_samp samples (from yaml: 64600)
NB_SAMP = 64600

# Map folder names to multi-class label indices
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


class Dataset_LibriSeVoc(Dataset):
    def __init__(self, file_list, label_binary, label_multi):
        """
        Args:
            file_list    : list of audio file paths
            label_binary : list of binary labels (0=real/gt, 1=fake)
            label_multi  : list of multi-class labels (0-6, folder index)
        """
        self.file_list     = file_list
        self.label_binary  = label_binary
        self.label_multi   = label_multi
        self.cut           = NB_SAMP

    def __len__(self):
        return len(self.file_list)

    def load_waveform(self, audio_path):
        """
        Load a WAV file using scipy first, falling back to librosa.
        Returns None if the file is empty/corrupt.
        """
        # Skip zero-length or tiny files immediately
        if os.path.getsize(audio_path) < 44:  # 44 bytes = minimum WAV header
            return None

        y = None
        sr = None

        # --- attempt 1: scipy ---
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

        # --- attempt 2: librosa fallback ---
        if y is None:
            try:
                y, sr = librosa.load(audio_path, sr=None, mono=True)
            except Exception:
                return None  # give up, caller will skip this sample

        if y is None or len(y) == 0:
            return None

        # Resample if needed
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Pad or trim to exactly NB_SAMP
        if len(y) < self.cut:
            num_repeats = (self.cut // len(y)) + 1
            y = np.tile(y, num_repeats)
        y = y[:self.cut]

        return Tensor(y)  # shape: (NB_SAMP,)

    def __getitem__(self, index):
        waveform     = self.load_waveform(self.file_list[index])
        label_bin    = self.label_binary[index]
        label_multi  = self.label_multi[index]
        return waveform, label_bin, label_multi


def collate_skip_none(batch):
    """Drop any samples where load_waveform returned None (corrupt/empty files)."""
    batch = [(w, b, m) for w, b, m in batch if w is not None]
    if len(batch) == 0:
        return None, None, None
    waveforms   = torch.stack([x[0] for x in batch])
    labels_bin  = torch.tensor([x[1] for x in batch], dtype=torch.long)
    labels_multi = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return waveforms, labels_bin, labels_multi


def build_file_lists(dataset_path, train_ratio=0.7, dev_ratio=0.15, seed=42):
    """
    Walk the LibriSeVoc folder structure and split into train/dev/test.

    Structure expected:
        dataset_path/
            gt/           *.wav   (real)
            diffwave/     *.wav   (fake)
            melgan/       *.wav   (fake)
            parallel_wave_gen/*.wav
            wavegrad/     *.wav
            wavenet/      *.wav
            wavernn/      *.wav
    """
    all_paths, all_binary, all_multi = [], [], []

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        folder_key = folder_name.lower()
        if folder_key not in FOLDER_TO_LABEL:
            print(f"  [WARN] Unknown folder '{folder_name}', skipping.")
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
        print(f"  {folder_name:20s}: {len(wav_files)} files  "
              f"(binary={binary_label}, multi={multi_label})")

    # Shuffle deterministically
    rng = random.Random(seed)
    combined = list(zip(all_paths, all_binary, all_multi))
    rng.shuffle(combined)
    all_paths, all_binary, all_multi = zip(*combined)

    n = len(all_paths)
    n_train = int(n * train_ratio)
    n_dev   = int(n * dev_ratio)

    splits = {
        'train': (all_paths[:n_train],
                  all_binary[:n_train],
                  all_multi[:n_train]),
        'dev':   (all_paths[n_train:n_train + n_dev],
                  all_binary[n_train:n_train + n_dev],
                  all_multi[n_train:n_train + n_dev]),
        'test':  (all_paths[n_train + n_dev:],
                  all_binary[n_train + n_dev:],
                  all_multi[n_train + n_dev:]),
    }

    for split, (paths, _, _) in splits.items():
        print(f"  {split:5s} split: {len(paths)} samples")

    return splits


def evaluate_accuracy(loader, model, device):
    num_correct_bin   = 0
    num_correct_multi = 0
    num_total         = 0
    model.eval()

    with torch.no_grad():
        for waveforms, labels_bin, labels_multi in loader:
            if waveforms is None:
                continue
            waveforms    = waveforms.to(device)
            labels_bin   = labels_bin.to(device)
            labels_multi = labels_multi.to(device)

            out_binary, out_multi = model(waveforms)

            _, preds_bin   = torch.max(out_binary, 1)
            _, preds_multi = torch.max(out_multi,  1)

            num_correct_bin   += (preds_bin   == labels_bin).sum().item()
            num_correct_multi += (preds_multi == labels_multi).sum().item()
            num_total         += labels_bin.size(0)

    acc_bin   = (num_correct_bin   / num_total) * 100
    acc_multi = (num_correct_multi / num_total) * 100
    return acc_bin, acc_multi


def train_epoch(train_loader, model, optimizer, device, criterion):
    model.train()
    running_loss      = 0.0
    num_correct_bin   = 0
    num_correct_multi = 0
    num_total         = 0

    for waveforms, labels_bin, labels_multi in tqdm(train_loader, total=len(train_loader)):
        if waveforms is None:
            continue
        waveforms    = waveforms.to(device)
        labels_bin   = labels_bin.to(device)
        labels_multi = labels_multi.to(device)

        optimizer.zero_grad()
        out_binary, out_multi = model(waveforms)

        # Combined loss: binary + multi-class
        loss = criterion(out_binary, labels_bin) + criterion(out_multi, labels_multi)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)

        _, preds_bin   = torch.max(out_binary, 1)
        _, preds_multi = torch.max(out_multi,  1)

        num_correct_bin   += (preds_bin   == labels_bin).sum().item()
        num_correct_multi += (preds_multi == labels_multi).sum().item()
        num_total         += labels_bin.size(0)

    acc_bin   = (num_correct_bin   / num_total) * 100
    acc_multi = (num_correct_multi / num_total) * 100
    avg_loss  = running_loss / num_total
    return avg_loss, acc_bin, acc_multi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',       type=str,   required=True,
                        help='Path to LibriSeVoc dataset root')
    parser.add_argument('--model_save_path', type=str,   default='./models')
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--num_epochs',      type=int,   default=50)
    parser.add_argument('--lr',              type=float, default=0.0001)
    parser.add_argument('--train_ratio',     type=float, default=0.70)
    parser.add_argument('--dev_ratio',       type=float, default=0.15)
    # remaining (1 - train - dev) goes to test
    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # ── Build file lists ──────────────────────────────────────────────────────
    print('\nScanning dataset...')
    splits = build_file_lists(
        args.data_path,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    train_set = Dataset_LibriSeVoc(*splits['train'])
    dev_set   = Dataset_LibriSeVoc(*splits['dev'])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=collate_skip_none)
    dev_loader   = DataLoader(dev_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_skip_none)

    # ── Load model ────────────────────────────────────────────────────────────
    with open('model_config_RawNet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model     = RawNet(config['model'], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # NLLLoss pairs with LogSoftmax inside the model
    criterion = nn.NLLLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        train_loss, train_acc_bin, train_acc_multi = train_epoch(
            train_loader, model, optimizer, device, criterion)

        val_acc_bin, val_acc_multi = evaluate_accuracy(dev_loader, model, device)

        print(
            f'Epoch {epoch+1:3d}/{args.num_epochs} | '
            f'Loss: {train_loss:.4f} | '
            f'Train bin/multi: {train_acc_bin:.1f}% / {train_acc_multi:.1f}% | '
            f'Val   bin/multi: {val_acc_bin:.1f}%  / {val_acc_multi:.1f}%'
        )

        if val_acc_bin > best_acc:
            best_acc = val_acc_bin
            save_path = os.path.join(args.model_save_path, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f'  ✔ Best model saved  (val binary acc: {best_acc:.2f}%)')

    print(f'\nTraining complete. Best val binary accuracy: {best_acc:.2f}%')