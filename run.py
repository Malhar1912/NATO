"""
rna_tm_nato_real.py

NATO-OPT training for Stanford RNA 3D Folding (Real Data)
Optimizes TM-score using the REAL Kaggle dataset.

Uses DistributedDataParallel (DDP) across 4 GPUs.

Fixes:
- Explicitly uses 'target_id' for sequences.
- Explicitly uses 'ID' for labels.
- Explicitly uses 'x_1', 'y_1', 'z_1' for coordinates.
- Filters sequences > 1000 length to prevent OOM.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Robust import for nato_opt — ensure the project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from nato_opt import NATOOptimizer

# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_DIR = r"C:\Users\BlockChain\.kaggle"

# Handle subfolders if extracted differently
if not os.path.exists(os.path.join(DATA_DIR, "train_sequences.csv")):
    SUB_DIR = os.path.join(DATA_DIR, "stanford-rna-3d-folding-2")
    if os.path.exists(SUB_DIR):
        DATA_DIR = SUB_DIR

EPOCHS = 10
LR = 5e-4
MAX_SEQ_LEN = 1000  # Filter out extremely long sequences (EDA showed max 125k!)

WORLD_SIZE = 4       # Number of GPUs to use
BATCH_SIZE = 8       # Total batch size (each GPU gets BATCH_SIZE // WORLD_SIZE = 2)

# --------------------------------------------------
# Dataset
# --------------------------------------------------
class RNADataset(Dataset):
    def __init__(self, seq_csv, label_csv, is_validation=False):
        print(f"Loading {seq_csv}...")
        self.seq_df = pd.read_csv(seq_csv)
       
        # Filter out massive sequences to prevent OOM
        initial_len = len(self.seq_df)
        self.seq_df = self.seq_df[self.seq_df['sequence'].str.len() <= MAX_SEQ_LEN]
        print(f"  Filtered {initial_len - len(self.seq_df)} sequences > {MAX_SEQ_LEN} length.")

        print(f"Loading {label_csv}...")
        # Load specific columns to save RAM
        # EDA Confirmed: ID, resid, x_1, y_1, z_1
        cols_to_load = ["ID", "resid", "x_1", "y_1", "z_1"]
        self.lbl_df = pd.read_csv(label_csv, usecols=cols_to_load, low_memory=False)
       
        # Drop rows with missing coordinates (EDA showed ~486k missing)
        self.lbl_df = self.lbl_df.dropna(subset=["x_1", "y_1", "z_1"])

        self.vocab = {"A": 0, "U": 1, "G": 2, "C": 3}

        print("Grouping labels by ID...")
        # Ensure IDs are strings
        self.seq_df['target_id'] = self.seq_df['target_id'].astype(str)
        self.lbl_df['ID'] = self.lbl_df['ID'].astype(str)
       
        # Filter labels to only include IDs present in our filtered sequence file
        valid_ids = set(self.seq_df["target_id"].unique())
        self.lbl_df = self.lbl_df[self.lbl_df["ID"].isin(valid_ids)]

        self.groups = {
            k: v.sort_values("resid")
            for k, v in self.lbl_df.groupby("ID")
        }

    def encode(self, seq):
        return torch.tensor([self.vocab.get(c, 0) for c in seq], dtype=torch.long)

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
       
        # EDA Confirmed: Sequence file uses 'target_id'
        seq_id = str(row["target_id"])
        seq = row["sequence"]

        if seq_id in self.groups:
            # EDA Confirmed: Label file uses 'x_1', 'y_1', 'z_1'
            coords = self.groups[seq_id][["x_1", "y_1", "z_1"]].values
        else:
            coords = np.zeros((len(seq), 3))

        # Truncate to matching length
        L = min(len(seq), len(coords))
       
        return (
            self.encode(seq[:L]),
            torch.tensor(coords[:L], dtype=torch.float32),
        )

# --------------------------------------------------
# Collate — device-agnostic (moved to CPU; each rank
#   transfers to its own GPU inside the training loop)
# --------------------------------------------------
def collate(batch):
    # Filter out empty items
    batch = [b for b in batch if b[0].shape[0] > 0]
    if not batch:
        return None

    seqs, coords = zip(*batch)
    max_len = max(len(s) for s in seqs)

    X, Y, mask = [], [], []

    for s, c in zip(seqs, coords):
        pad = max_len - len(s)
        X.append(F.pad(s, (0, pad)))
        Y.append(F.pad(c, (0, 0, 0, pad)))
        mask.append([1] * len(s) + [0] * pad)

    return (
        torch.stack(X),
        torch.stack(Y),
        torch.tensor(mask, dtype=torch.float32),
    )

# --------------------------------------------------
# Model
# --------------------------------------------------
class RNATransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, layers=4):
        super().__init__()
        self.embed = nn.Embedding(4, d_model)
        # Positional encoding large enough for MAX_SEQ_LEN
        self.pos_enc = nn.Parameter(torch.randn(1, MAX_SEQ_LEN + 50, d_model))
       
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.out = nn.Linear(d_model, 3)

    def forward(self, x):
        B, L = x.shape
        x = self.embed(x)
        if L <= self.pos_enc.shape[1]:
            x = x + self.pos_enc[:, :L, :]
        else:
            x = x + F.pad(self.pos_enc, (0, 0, 0, L - self.pos_enc.shape[1]))
           
        x = self.encoder(x)
        return self.out(x)

# --------------------------------------------------
# TM-score utilities
# --------------------------------------------------
def d0(Lref):
    if Lref < 12: return 0.3
    if Lref < 16: return 0.4
    if Lref < 20: return 0.5
    if Lref < 24: return 0.6
    if Lref < 30: return 0.7
    return 1.24 * (Lref - 15) ** (1 / 3) - 1.8

def kabsch(P, Q):
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    C = P.T @ Q
    try:
        V, S, W = torch.svd(C)
    except RuntimeError:
        return P, Q
    d = torch.det(V @ W.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device))
    U = V @ D @ W.T
    return P @ U, Q

def tm_score(pred, true, mask):
    scores = []
    for p, t, m in zip(pred, true, mask):
        idx = m.bool()
        p = p[idx]
        t = t[idx]
        L = len(p)
        if L < 3:
            scores.append(torch.tensor(0.0, device=p.device))
            continue
        p_aligned, t_aligned = kabsch(p, t)
        di = torch.norm(p_aligned - t_aligned, dim=1)
        d0_i = d0(L)
        score = torch.mean(1.0 / (1.0 + (di / d0_i) ** 2))
        scores.append(score)
   
    if not scores: return torch.tensor(0.0, device=pred.device)
    return torch.stack(scores).mean()

# --------------------------------------------------
# DDP setup / cleanup
# --------------------------------------------------
def setup_ddp(rank, world_size):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Destroy the distributed process group."""
    dist.destroy_process_group()

# --------------------------------------------------
# Training loop
# --------------------------------------------------
def train_epoch(model, optimizer, loader, device, rank):
    model.train()
    tms = []

    for i, batch in enumerate(loader):
        if batch is None:
            continue
        seq, coords, mask = batch
        seq = seq.to(device)
        coords = coords.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
       
        pred = model(seq)
        tm = tm_score(pred, coords, mask)
        loss = 1.0 - tm

        loss.backward()
        optimizer.step()

        tms.append(tm.item())
       
        if rank == 0 and i % 10 == 0:
            print(f"  Batch {i}: TM-Score = {tm.item():.4f}")

    return float(np.mean(tms)) if tms else 0.0

def eval_epoch(model, loader, device):
    model.eval()
    tms = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            seq, coords, mask = batch
            seq = seq.to(device)
            coords = coords.to(device)
            mask = mask.to(device)

            pred = model(seq)
            tm = tm_score(pred, coords, mask)
            tms.append(tm.item())
    return float(np.mean(tms)) if tms else 0.0

# --------------------------------------------------
# Main worker (one per GPU)
# --------------------------------------------------
def main_worker(rank, world_size):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"DDP launched with {world_size} GPUs")
        print(f"Total batch size: {BATCH_SIZE}  |  Per-GPU batch size: {BATCH_SIZE // world_size}")
        print(f"Data Directory: {DATA_DIR}")

    train_seq_path = os.path.join(DATA_DIR, "train_sequences.csv")
    train_lbl_path = os.path.join(DATA_DIR, "train_labels.csv")
   
    if not os.path.exists(train_seq_path):
        if rank == 0:
            print(f"ERROR: Could not find {train_seq_path}")
        cleanup_ddp()
        return

    # Load datasets (every rank loads the data; DistributedSampler
    # ensures each rank processes a unique shard)
    train_ds = RNADataset(train_seq_path, train_lbl_path)

    val_seq_path = os.path.join(DATA_DIR, "validation_sequences.csv")
    val_lbl_path = os.path.join(DATA_DIR, "validation_labels.csv")

    if os.path.exists(val_seq_path) and os.path.exists(val_lbl_path):
        val_ds = RNADataset(val_seq_path, val_lbl_path, is_validation=True)
    else:
        if rank == 0:
            print("Validation files not found, using subset of train.")
        val_ds = train_ds

    # DistributedSampler assigns a unique subset to each rank
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    per_gpu_batch = BATCH_SIZE // world_size  # 8 // 4 = 2

    # Windows: num_workers=0 is required for CUDA stability
    train_loader = DataLoader(
        train_ds, batch_size=per_gpu_batch, sampler=train_sampler,
        collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=per_gpu_batch, sampler=val_sampler,
        collate_fn=collate, num_workers=0
    )

    # Build model and wrap with DDP
    model = RNATransformer().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = NATOOptimizer(model.parameters(), lr=LR)

    if rank == 0:
        print("Starting Training...")

    for epoch in range(1, EPOCHS + 1):
        # Set epoch on sampler so shuffling differs each epoch
        train_sampler.set_epoch(epoch)

        train_tm = train_epoch(model, optimizer, train_loader, device, rank)
        val_tm = eval_epoch(model, val_loader, device)

        if rank == 0:
            print(f"Epoch {epoch:02d} | Train TM {train_tm:.4f} | Val TM {val_tm:.4f}")

    cleanup_ddp()

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    mp.spawn(main_worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)