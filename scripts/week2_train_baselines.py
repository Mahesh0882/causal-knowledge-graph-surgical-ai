"""
Week 2: Train and Evaluate Baseline Models
============================================
Trains Markov, LSTM, and Transformer baselines for next-action prediction.

Usage:
    python scripts/week2_train_baselines.py [--epochs 30] [--window 10] [--batch 64]
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.models import MarkovBaseline, LSTMPredictor, TransformerPredictor
from src.metrics import compute_metrics, print_metrics


# ---- Configuration ----
SEQUENCES_DIR = PROJECT_ROOT / "outputs" / "temporal_sequences"
SPLITS_PATH = PROJECT_ROOT / "outputs" / "data_splits.json"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "week2_results"


def get_device():
    """Select best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate model on a DataLoader. Returns (preds, targets) as numpy arrays."""
    model.eval()
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def train_neural_model(
    model,
    loaders,
    model_name,
    device,
    epochs=30,
    lr=1e-3,
    patience=5,
):
    """Full training loop with early stopping.

    Returns
    -------
    dict
        Best validation metrics.
    """
    print(f"\n{'#' * 60}")
    print(f"  Training: {model_name}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'#' * 60}")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = 0.0
    best_metrics = {}
    epochs_without_improvement = 0
    best_state = None

    checkpoint_dir = RESULTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)

        # Validate
        val_preds, val_targets = evaluate_model(model, loaders["val"], device)
        val_metrics = compute_metrics(val_preds, val_targets)
        val_f1 = val_metrics["f1_samples"]

        scheduler.step(val_f1)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Top-1: {val_metrics['top1_acc']:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val mAP: {val_metrics['mAP']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save best checkpoint
    if best_state is not None:
        ckpt_path = checkpoint_dir / f"{model_name.lower().replace(' ', '_')}_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"  Saved best checkpoint: {ckpt_path}")

        # Reload best weights for final evaluation
        model.load_state_dict(best_state)

    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Device: {device}")
    print(f"Window size: {args.window}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs: {args.epochs}")

    # Build dataloaders
    print("\nBuilding dataloaders...")
    loaders, num_classes = build_dataloaders(
        sequences_dir=SEQUENCES_DIR,
        splits_path=SPLITS_PATH,
        window_size=args.window,
        batch_size=args.batch,
    )
    print(f"Number of classes: {num_classes}")

    all_results = {}

    # ---- Baseline 1: Markov ----
    print("\n" + "=" * 60)
    print("  BASELINE 1: Markov (Transition Matrix)")
    print("=" * 60)

    markov = MarkovBaseline(num_classes=num_classes)
    markov.fit(loaders["train"])

    for split_name in ["val", "test"]:
        preds, targets = markov.predict_from_loader(loaders[split_name])
        metrics = compute_metrics(preds, targets)
        print_metrics(metrics, f"Markov ({split_name})")
        all_results[f"markov_{split_name}"] = metrics

    # ---- Baseline 2: LSTM ----
    lstm_model = LSTMPredictor(
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        embed_dim=64,
    )
    lstm_best = train_neural_model(
        lstm_model, loaders, "LSTM", device,
        epochs=args.epochs, lr=1e-3, patience=7,
    )
    all_results["lstm_val"] = lstm_best

    # Evaluate on test
    test_preds, test_targets = evaluate_model(lstm_model, loaders["test"], device)
    lstm_test = compute_metrics(test_preds, test_targets)
    print_metrics(lstm_test, "LSTM (test)")
    all_results["lstm_test"] = lstm_test

    # ---- Baseline 3: Transformer ----
    transformer_model = TransformerPredictor(
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
    )
    tf_best = train_neural_model(
        transformer_model, loaders, "Transformer", device,
        epochs=args.epochs, lr=5e-4, patience=7,
    )
    all_results["transformer_val"] = tf_best

    # Evaluate on test
    test_preds, test_targets = evaluate_model(transformer_model, loaders["test"], device)
    tf_test = compute_metrics(test_preds, test_targets)
    print_metrics(tf_test, "Transformer (test)")
    all_results["transformer_test"] = tf_test

    # ---- Save all results ----
    results_path = RESULTS_DIR / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved: {results_path}")

    # ---- Print comparison table ----
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON (Test Set)")
    print("=" * 70)
    header = f"{'Model':<15} {'Top-1':>8} {'Top-5':>8} {'F1-samp':>8} {'F1-mac':>8} {'mAP':>8}"
    print(header)
    print("-" * 70)
    for name, key in [("Markov", "markov_test"), ("LSTM", "lstm_test"), ("Transformer", "transformer_test")]:
        m = all_results[key]
        print(f"{name:<15} {m['top1_acc']:>8.4f} {m['top5_acc']:>8.4f} {m['f1_samples']:>8.4f} {m['f1_macro']:>8.4f} {m['mAP']:>8.4f}")
    print("=" * 70)

    # Save config
    config = {
        "window_size": args.window,
        "batch_size": args.batch,
        "epochs": args.epochs,
        "device": str(device),
        "num_classes": num_classes,
        "train_samples": len(loaders["train"].dataset),
        "val_samples": len(loaders["val"].dataset),
        "test_samples": len(loaders["test"].dataset),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models for next-action prediction")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--window", type=int, default=10, help="Input window size (frames)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    main(args)
