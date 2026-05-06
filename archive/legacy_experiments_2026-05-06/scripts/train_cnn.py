"""
Fix G: Train and evaluate CNN baseline on ASVspoof 2019 LA.

Uses log-mel spectrograms, fixed-length segments (3s), BCEWithLogitsLoss.
Saves best checkpoint by dev EER.

Usage:
    python scripts/train_cnn.py --data data/LA
    python scripts/train_cnn.py --data data/LA --epochs 20 --batch 32
"""

import argparse
import os
import sys

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_split, split_summary
from src.cnn import build_model

SR        = 16000
N_MELS    = 64
CLIP_SECS = 3       # fixed clip length — pad/trim all utterances to this
HOP       = 160     # 10 ms


def audio_to_logmel(audio: np.ndarray, sr: int = SR,
                    clip_secs: float = CLIP_SECS) -> np.ndarray:
    target = clip_secs * sr
    if len(audio) < target:
        audio = np.pad(audio, (0, int(target) - len(audio)))
    else:
        audio = audio[:int(target)]
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                         hop_length=HOP, fmax=8000)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # [N_MELS, T]


class ASVDataset(Dataset):
    def __init__(self, samples, clip_secs=CLIP_SECS):
        self.samples   = samples
        self.clip_secs = clip_secs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio, sr = librosa.load(s.path, sr=SR, mono=True)
        spec = audio_to_logmel(audio, sr, self.clip_secs)   # [N_MELS, T]
        x = torch.from_numpy(spec).unsqueeze(0)             # [1, N_MELS, T]
        y = torch.tensor(float(s.label))                    # bonafide=1, spoof=0
        return x, y


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0.0, 1.0)
    return float(eer)


def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores.tolist())
            all_labels.extend(y.numpy().tolist())
    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)
    eer = compute_eer(y_true, y_scores)
    acc = float(((y_scores > 0.5).astype(int) == y_true).mean())
    return eer, acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_samples = load_split(args.data, "train", args.limit)
    dev_samples   = load_split(args.data, "dev",   args.limit)
    print(f"Train: {split_summary(train_samples)}")
    print(f"Dev  : {split_summary(dev_samples)}")

    train_loader = DataLoader(ASVDataset(train_samples), batch_size=args.batch,
                              shuffle=True,  num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(ASVDataset(dev_samples),   batch_size=args.batch,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = build_model(n_mels=N_MELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.output, exist_ok=True)
    best_eer  = 1.0
    best_path = os.path.join(args.output, "cnn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        eer, acc = evaluate(model, dev_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:2d} | loss={avg_loss:.4f} | dev EER={eer*100:.2f}% | acc={acc*100:.2f}%")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (EER={best_eer*100:.2f}%)")

    print(f"\nTraining complete. Best dev EER: {best_eer*100:.2f}%")
    print(f"Checkpoint: {best_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/LA")
    parser.add_argument("--output", default="results/models")
    parser.add_argument("--epochs", type=int,   default=15)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--limit",  type=int,   default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
