#!/usr/bin/env python3
from __future__ import annotations
"""
Minimal TCN training scaffold for state segmentation on exported features.
Requires: torch, torchmetrics (optional). Install before running.
Usage:
  python scripts/train_tcn.py --features path/to/features.parquet --out runs/tcn
"""
import argparse, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Features file (parquet or csv) from export_features.py")
    ap.add_argument("--out", required=True, help="Output dir")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except Exception as e:
        print("Install torch to use this script: pip install torch")
        print("Error:", e)
        return

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.features) if args.features.endswith(".parquet") else pd.read_csv(args.features)

    # Example labels from speed thresholds (weak labels) — replace with GT if available
    speed = df["speed_mps"].values
    labels = np.where(speed > 0.5, 2, np.where(speed > 0.08, 1, 0)).astype(np.int64)
    features = df[[
        "speed_mps", "acc_mps2", "dir_change_rad", "w", "h", "area", "nearest_pallet_dist_m"
    ]].fillna(0.0).values.astype(np.float32)

    # Group by actor for sequences
    seqs = []
    lbls = []
    for aid, g in df.groupby("actor_id"):
        seqs.append(features[g.index])
        lbls.append(labels[g.index])

    # Pad/truncate to fixed length for batching (simple baseline)
    T = 256
    D = features.shape[1]

    def pad_seq(x, t=T):
        if len(x) >= t:
            return x[:t]
        out = np.zeros((t, x.shape[1]), dtype=np.float32)
        out[:len(x)] = x
        return out

    X = np.stack([pad_seq(x) for x in seqs], axis=0)  # [N, T, D]
    Y = np.stack([pad_seq(y[:, None]).squeeze(-1) for y in lbls], axis=0)  # [N, T]

    class SimpleTCN(nn.Module):
        def __init__(self, d_in, d_hid=64, n_classes=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(d_in, d_hid, 5, padding=2), nn.ReLU(),
                nn.Conv1d(d_hid, d_hid, 5, padding=2), nn.ReLU(),
                nn.Conv1d(d_hid, n_classes, 1)
            )
        def forward(self, x):  # x: [B, T, D]
            x = x.transpose(1, 2)  # [B, D, T]
            logits = self.net(x)   # [B, C, T]
            return logits.transpose(1, 2)  # [B, T, C]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTCN(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_t)
        loss = crit(logits.reshape(-1, logits.shape[-1]), Y_t.reshape(-1))
        loss.backward()
        opt.step()
        print(f"Epoch {ep+1}/{args.epochs} loss={loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(args.out, "tcn.pt"))
    print("Saved model to", os.path.join(args.out, "tcn.pt"))


if __name__ == "__main__":
    main()

