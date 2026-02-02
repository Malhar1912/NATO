"""
nsl_kdd_nato_experiment.py

Run multiple trials comparing NATO-OPT vs Adam on NSL-KDD (tabular data),
compute p-values, effect sizes, and save results.csv

Requirements:
    pip install pandas sklearn torch scipy nato_opt
"""

import os
import random
import csv
from statistics import mean, stdev

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from scipy.stats import ttest_rel, wilcoxon

# NATO optimizer utilities
from nato_opt import NATOOptimizer, fourier_spectral_penalty, low_pass_filter_gradients


# -----------------------------------------------------
# Experiment config
# -----------------------------------------------------
N_TRIALS = 10
EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_FSP = 1e-6
CUTOFF_RATIO = 0.5
RESULTS_CSV = "results_nsl_kdd.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------------------------------
# Download + Load NSL-KDD
# NSL-KDD contains:
#    KDDTrain+.txt     (train)
#    KDDTest+.txt      (test)
#
# Direct mirror used: https://github.com/defcom17/NSL_KDD
# -----------------------------------------------------
def download_nsl_kdd():
    if not os.path.exists("KDDTrain+.txt"):
        os.system("wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt")
    if not os.path.exists("KDDTest+.txt"):
        os.system("wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt")

download_nsl_kdd()


# NSL-KDD column names
COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]


# -----------------------------------------------------
# Load + preprocess
# -----------------------------------------------------
def load_preprocessed_data():
    train_df = pd.read_csv("KDDTrain+.txt", names=COLS)
    test_df  = pd.read_csv("KDDTest+.txt",  names=COLS)

    # Binary classification: "normal" vs "attack"
    train_df["label"] = train_df["label"].apply(lambda x: 0 if x == "normal" else 1)
    test_df["label"]  = test_df["label"].apply(lambda x: 0 if x == "normal" else 1)

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"].values

    X_test  = test_df.drop("label", axis=1)
    y_test  = test_df["label"].values

    # Identify categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]

    # numeric
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans  = preprocessor.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_trans.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor  = torch.tensor(X_test_trans.toarray(), dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)


(X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor) = load_preprocessed_data()


def make_loaders(seed):
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    test_ds  = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# -----------------------------------------------------
# MLP Model for NSL-KDD
# -----------------------------------------------------
INPUT_DIM = X_train_tensor.shape[1]

class NSLKDDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


criterion = nn.CrossEntropyLoss()


# -----------------------------------------------------
# Training + evaluation
# -----------------------------------------------------
def train_one_epoch(model, optimizer, train_loader, epoch):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        ce_loss = criterion(outputs, targets)

        try:
            fsp = fourier_spectral_penalty(model, lambda_fsp=LAMBDA_FSP)
        except:
            fsp = 0.0

        loss = ce_loss + fsp
        loss.backward()

        try:
            low_pass_filter_gradients(model, cutoff_ratio=CUTOFF_RATIO)
        except:
            pass

        try:
            optimizer.step(epoch)
        except TypeError:
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)


def evaluate_accuracy(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


# -----------------------------------------------------
# Trial runner
# -----------------------------------------------------
def run_trial(seed, optimizer_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, test_loader = make_loaders(seed)
    model = NSLKDDModel().to(DEVICE)

    if optimizer_name.lower() == "nato":
        optimizer = NATOOptimizer(model.parameters(), lr=LR)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        raise ValueError("Choose nato or adam")

    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_loader, epoch)

    return evaluate_accuracy(model, test_loader)


# -----------------------------------------------------
# Main experiment loop
# -----------------------------------------------------
def run_experiment():
    seeds = [1234 + i * 101 for i in range(N_TRIALS)]
    nato_accs, adam_accs = [], []

    for i, seed in enumerate(seeds, 1):
        print(f"\nTrial {i}/{N_TRIALS} (seed={seed}) - NATO")
        nato_acc = run_trial(seed, "nato")
        print(f" NATO accuracy: {nato_acc:.4f}")
        nato_accs.append(nato_acc)

        print(f"Trial {i}/{N_TRIALS} (seed={seed}) - Adam")
        adam_acc = run_trial(seed, "adam")
        print(f" Adam accuracy: {adam_acc:.4f}")
        adam_accs.append(adam_acc)

    # Statistics
    nato_mean, nato_std = mean(nato_accs), stdev(nato_accs)
    adam_mean, adam_std = mean(adam_accs), stdev(adam_accs)

    t_stat, p_val_t = ttest_rel(nato_accs, adam_accs)

    try:
        w_stat, p_val_w = wilcoxon(np.array(nato_accs) - np.array(adam_accs))
    except:
        w_stat, p_val_w = None, None

    diffs = np.array(nato_accs) - np.array(adam_accs)
    d_mean = diffs.mean()
    d_std = diffs.std(ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else float("nan")

    print("\n=== NSL-KDD Experiment Summary ===")
    print(f"NATO mean acc: {nato_mean:.4f} ± {nato_std:.4f}")
    print(f"Adam mean acc: {adam_mean:.4f} ± {adam_std:.4f}")
    print(f"Mean diff (NATO - Adam): {d_mean:.6f}")
    print(f"Cohen's d: {cohens_d:.4f}")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val_t:.4e}")
    if p_val_w is not None:
        print(f"Wilcoxon: W={w_stat}, p={p_val_w:.4e}")

    # Save CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "seed", "nato_acc", "adam_acc", "diff"])
        for i, seed in enumerate(seeds):
            writer.writerow([i+1, seed, nato_accs[i], adam_accs[i], nato_accs[i] - adam_accs[i]])

    print(f"Saved results to {RESULTS_CSV}")

    return {
        "nato_accs": nato_accs,
        "adam_accs": adam_accs,
        "t_stat": t_stat,
        "p_val_t": p_val_t,
        "wilcoxon_stat": w_stat,
        "p_val_w": p_val_w,
        "cohens_d": cohens_d,
    }


if __name__ == "__main__":
    results = run_experiment()
