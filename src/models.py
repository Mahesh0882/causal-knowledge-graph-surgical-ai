"""
Baseline Models for Next-Action Prediction
============================================
1. MarkovBaseline: statistical transition matrix baseline
2. LSTMPredictor: LSTM-based sequence model
3. TransformerPredictor: Transformer encoder-based model
"""

import json
import numpy as np
import torch
import torch.nn as nn
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Baseline 1: Markov (Transition Matrix)
# ---------------------------------------------------------------------------

class MarkovBaseline:
    """Statistical baseline using the global transition matrix.

    For each current multi-hot frame, computes P(next triplet class)
    by averaging the transition probabilities from all active classes.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.transition_counts = np.zeros((num_classes, num_classes), dtype=np.float64)

    def fit(self, dataloader):
        """Build transition matrix from training data."""
        for x_batch, y_batch in dataloader:
            # x_batch: (B, window, C), y_batch: (B, C)
            # Use only the LAST frame in the window as source
            last_frame = x_batch[:, -1, :].numpy()  # (B, C)
            next_frame = y_batch.numpy()  # (B, C)

            for b in range(last_frame.shape[0]):
                src_active = np.where(last_frame[b] > 0)[0]
                tgt_active = np.where(next_frame[b] > 0)[0]
                for s in src_active:
                    for t in tgt_active:
                        self.transition_counts[s, t] += 1

        # Normalize rows to probabilities
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)  # avoid division by zero
        self.transition_probs = self.transition_counts / row_sums

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict next-frame probabilities.

        Parameters
        ----------
        x : np.ndarray, shape (B, window, C) or (B, C)
            Input sequence. Only the last frame is used.

        Returns
        -------
        np.ndarray, shape (B, C)
            Predicted probabilities for each class.
        """
        if x.ndim == 3:
            x = x[:, -1, :]  # use last frame only

        preds = np.zeros_like(x, dtype=np.float64)
        for b in range(x.shape[0]):
            active = np.where(x[b] > 0)[0]
            if len(active) > 0:
                # Average transition probabilities from all active classes
                preds[b] = self.transition_probs[active].mean(axis=0)
            else:
                # Uniform prediction if no active classes
                preds[b] = 1.0 / self.num_classes
        return preds

    def predict_from_loader(self, dataloader):
        """Generate predictions and targets for an entire DataLoader."""
        all_preds = []
        all_targets = []
        for x_batch, y_batch in dataloader:
            preds = self.predict(x_batch.numpy())
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
        return np.concatenate(all_preds), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Baseline 2: LSTM
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """LSTM-based next-action predictor.

    Architecture:
        multi-hot input -> linear projection -> LSTM -> FC -> sigmoid

    Parameters
    ----------
    num_classes : int
        Number of triplet classes (size of multi-hot vector).
    hidden_dim : int
        LSTM hidden dimension.
    num_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate (applied between LSTM layers).
    embed_dim : int
        Input projection dimension.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Project multi-hot to dense embedding
        self.input_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, C)
            Multi-hot sequence of T frames with C classes.

        Returns
        -------
        Tensor, shape (B, C)
            Logits for each class in the next frame.
        """
        # Project input
        x = self.input_proj(x)  # (B, T, embed_dim)

        # LSTM
        out, _ = self.lstm(x)  # (B, T, hidden_dim)
        last_hidden = out[:, -1, :]  # (B, hidden_dim)

        # Predict
        logits = self.head(last_hidden)  # (B, C)
        return logits


# ---------------------------------------------------------------------------
# Baseline 3: Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer encoder-based next-action predictor.

    Architecture:
        multi-hot -> linear projection -> positional encoding
        -> Transformer encoder -> pool last token -> FC -> logits

    Parameters
    ----------
    num_classes : int
        Number of triplet classes.
    d_model : int
        Transformer model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of Transformer encoder layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(num_classes, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, C)

        Returns
        -------
        Tensor, shape (B, C)
        """
        # Causal mask: each position can only attend to itself and previous
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask=causal_mask)  # (B, T, d_model)
        x = x[:, -1, :]  # last position (B, d_model)
        logits = self.head(x)  # (B, C)
        return logits
