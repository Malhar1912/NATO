"""
yahoo_noise_nato_experiment.py

Run multiple trials comparing NATO-OPT vs Adam on
Yahoo Answers Topic Classification (noisy training).

Each trial injects new random noise → trains 1 epoch →
evaluates accuracy + macro-F1 → records stats.

Requirements:
    pip install transformers datasets torch scipy nato_opt nltk scikit-learn
    python -c "import nltk; nltk.download('wordnet')"
"""

import os
import random
import csv
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator
)

from nltk.corpus import wordnet
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import f1_score

# NATO optimizer
from nato_opt import NATOOptimizer, fourier_spectral_penalty, low_pass_filter_gradients

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
LR = 3e-5
EPOCHS = 1
BATCH_SIZE = 16
N_TRIALS = 5
RESULTS_CSV = "results_yahoo_noise.csv"

NOISE_LEVEL = 0.15  # fraction of tokens modified

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)


# --------------------------------------------------
# TEXT NOISE FUNCTIONS (same as SQuAD version)
# --------------------------------------------------
def random_char_swap(text):
    chars = list(text)
    if len(chars) > 1:
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)


def random_word_delete(words):
    if len(words) < 2:
        return words
    idx = random.randint(0, len(words)-1)
    return words[:idx] + words[idx+1:]


def synonym_replacement(words):
    idx = random.randint(0, len(words)-1)
    syns = wordnet.synsets(words[idx])
    if syns:
        lemmas = syns[0].lemma_names()
        if lemmas:
            words[idx] = lemmas[0].replace("_", " ")
    return words


def apply_noise(text, noise_level=NOISE_LEVEL):
    words = text.split()
    if len(words) == 0:
        return text

    n_changes = max(1, int(len(words) * noise_level))

    for _ in range(n_changes):
        r = random.random()
        if r < 0.33:
            text = random_char_swap(text)
        elif r < 0.66:
            words = random_word_delete(words)
            text = " ".join(words)
        else:
            words = synonym_replacement(words)
            text = " ".join(words)

    return text


# --------------------------------------------------
# LOAD YAHOO + INJECT NOISE
# --------------------------------------------------
def load_yahoo_with_noise():
    ds = load_dataset("yahoo_answers_topics")

    def add_noise(example):
        noisy_q = apply_noise(example["question_title"])
        noisy_c = apply_noise(example["question_content"])
        example["text"] = noisy_q + " " + noisy_c
        return example

    noisy_train = ds["train"].map(add_noise)
    noisy_test  = ds["test"].map(add_noise)
    return noisy_train, noisy_test


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )


# --------------------------------------------------
# TRAIN + EVAL
# --------------------------------------------------
def run_trial(seed, optimizer_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ds, test_ds = load_yahoo_with_noise()

    train_ds = train_ds.map(preprocess_function, batched=True)
    test_ds  = test_ds.map(preprocess_function, batched=True)

    train_ds.set_format("torch")
    test_ds.set_format("torch")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=default_data_collator)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=default_data_collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=10
    ).to(DEVICE)

    if optimizer_name.lower() == "nato":
        optimizer = NATOOptimizer(model.parameters(), lr=LR)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---- TRAIN ----
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )

            loss = outputs.loss

            # spectral regularization
            try:
                fsp = fourier_spectral_penalty(model, lambda_fsp=1e-6)
                loss = loss + fsp
            except:
                pass

            loss.backward()

            # low-pass filter gradients
            try:
                low_pass_filter_gradients(model, cutoff_ratio=0.5)
            except:
                pass

            # NATO uses step(epoch), Adam uses step()
            try:
                optimizer.step(epoch)
            except:
                optimizer.step()

            optimizer.zero_grad()

    # ---- EVAL ----
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )

            p = outputs.logits.argmax(-1).cpu().numpy()
            y = batch["label"].cpu().numpy()

            preds.extend(p.tolist())
            labels.extend(y.tolist())

    acc = (np.array(preds) == np.array(labels)).mean()
    f1 = f1_score(labels, preds, average="macro")

    return acc, f1


# --------------------------------------------------
# EXPERIMENT LOOP
# --------------------------------------------------
def run_experiment():
    seeds = [5555 + i * 13 for i in range(N_TRIALS)]
    nato_acc, adam_acc = [], []
    nato_f1, adam_f1 = [], []

    for i, seed in enumerate(seeds, 1):
        print(f"\n--- Trial {i}/{N_TRIALS} seed={seed} NATO ---")
        acc, f1 = run_trial(seed, "nato")
        nato_acc.append(acc); nato_f1.append(f1)
        print(f" NATO ACC={acc:.4f}  F1={f1:.4f}")

        print(f"--- Trial {i}/{N_TRIALS} seed={seed} Adam ---")
        acc, f1 = run_trial(seed, "adam")
        adam_acc.append(acc); adam_f1.append(f1)
        print(f" Adam ACC={acc:.4f}  F1={f1:.4f}")

    # statistical tests
    acc_t, acc_p = ttest_rel(nato_acc, adam_acc)
    f1_t, f1_p = ttest_rel(nato_f1, adam_f1)

    try:
        acc_w, acc_pw = wilcoxon(np.array(nato_acc)-np.array(adam_acc))
        f1_w, f1_pw = wilcoxon(np.array(nato_f1)-np.array(adam_f1))
    except:
        acc_w = acc_pw = f1_w = f1_pw = None

    diff_acc = np.array(nato_acc) - np.array(adam_acc)
    diff_f1  = np.array(nato_f1) - np.array(adam_f1)

    d_acc = diff_acc.mean() / diff_acc.std(ddof=1)
    d_f1  = diff_f1.mean() / diff_f1.std(ddof=1)

    print("\n===== Yahoo Noise Results =====")
    print(f"NATO ACC: {mean(nato_acc):.4f} ± {stdev(nato_acc):.4f}")
    print(f"Adam ACC: {mean(adam_acc):.4f} ± {stdev(adam_acc):.4f}")
    print(f"NATO F1: {mean(nato_f1):.4f} ± {stdev(nato_f1):.4f}")
    print(f"Adam F1: {mean(adam_f1):.4f} ± {stdev(adam_f1):.4f}")

    print(f"Paired t-test ACC: t={acc_t:.4f}  p={acc_p:.4e}")
    print(f"Paired t-test F1:  t={f1_t:.4f}  p={f1_p:.4e}")
    print(f"Cohen's d ACC: {d_acc:.4f}")
    print(f"Cohen's d F1:  {d_f1:.4f}")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial","seed","nato_acc","adam_acc","nato_f1","adam_f1"])
        for i, s in enumerate(seeds):
            writer.writerow([i+1, s, nato_acc[i], adam_acc[i], nato_f1[i], adam_f1[i]])

    print(f"\nSaved trial results to {RESULTS_CSV}")


if __name__ == "__main__":
    run_experiment()
