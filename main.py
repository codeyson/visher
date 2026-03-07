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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from model import RawNet
import librosa
import scipy.io.wavfile as wavfile
from tqdm import tqdm

SAMPLE_RATE = 24000
NB_SAMP = 64600

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
        self.file_list     = file_list
        self.label_binary  = label_binary
        self.label_multi   = label_multi
        self.cut           = NB_SAMP

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
            num_repeats = (self.cut // len(y)) + 1
            y = np.tile(y, num_repeats)
        y = y[:self.cut]

        return Tensor(y)

    def __getitem__(self, index):
        waveform     = self.load_waveform(self.file_list[index])
        label_bin    = self.label_binary[index]
        label_multi  = self.label_multi[index]
        return waveform, label_bin, label_multi


def collate_skip_none(batch):
    batch = [(w, b, m) for w, b, m in batch if w is not None]
    if len(batch) == 0:
        return None, None, None
    waveforms    = torch.stack([x[0] for x in batch])
    labels_bin   = torch.tensor([x[1] for x in batch], dtype=torch.long)
    labels_multi = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return waveforms, labels_bin, labels_multi


def make_balanced_sampler(label_binary):
    """
    WeightedRandomSampler so every batch sees ~50% real and ~50% fake,
    regardless of the 6:1 imbalance in the dataset.
    """
    labels = list(label_binary)
    n_real = labels.count(0)
    n_fake = labels.count(1)
    w_real = 1.0 / n_real if n_real > 0 else 0
    w_fake = 1.0 / n_fake if n_fake > 0 else 0
    weights = [w_real if l == 0 else w_fake for l in labels]
    print(f'  Balanced sampler: {n_real} real, {n_fake} fake  '
          f'(w_real={w_real:.6f}, w_fake={w_fake:.6f})')
    return WeightedRandomSampler(weights=weights,
                                 num_samples=len(weights),
                                 replacement=True)


def build_file_lists(dataset_path, train_ratio=0.7, dev_ratio=0.15, seed=42):
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

    rng = random.Random(seed)
    combined = list(zip(all_paths, all_binary, all_multi))
    rng.shuffle(combined)
    all_paths, all_binary, all_multi = zip(*combined)

    n       = len(all_paths)
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


def train_epoch(train_loader, model, optimizer, device, criterion_bin, criterion_multi):
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

        loss = criterion_bin(out_binary, labels_bin) + criterion_multi(out_multi, labels_multi)
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

    # ── Balanced sampler: equal real/fake in every batch ─────────────────────
    print('\nBuilding balanced sampler...')
    sampler = make_balanced_sampler(splits['train'][1])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=sampler,
                              num_workers=4, collate_fn=collate_skip_none)
    dev_loader   = DataLoader(dev_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4,
                              collate_fn=collate_skip_none)

    # ── Load model ────────────────────────────────────────────────────────────
    with open('model_config_RawNet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model     = RawNet(config['model'], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Weighted loss: penalise misclassifying real voices more ──────────────
    binary_weights  = torch.tensor([6.0, 1.0]).to(device)
    multi_weights   = torch.tensor([6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion_bin   = nn.NLLLoss(weight=binary_weights)
    criterion_multi = nn.NLLLoss(weight=multi_weights)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(args.num_epochs):
        train_loss, train_acc_bin, train_acc_multi = train_epoch(
            train_loader, model, optimizer, device, criterion_bin, criterion_multi)

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