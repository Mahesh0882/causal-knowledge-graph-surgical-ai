"""
Surgical Triplet Sequence Dataset
==================================
PyTorch Dataset for next-action prediction from temporal triplet sequences.

Each sample is a window of k consecutive frames (multi-hot vectors),
with the target being the multi-hot vector of the next frame.
"""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union


class TripletSequenceDataset(Dataset):
    """Dataset for next-action prediction from surgical triplet sequences.

    Each sample consists of:
        - x: (window_size, num_classes) multi-hot tensor of active triplets
        - y: (num_classes,) multi-hot tensor of the NEXT frame's triplets
        - meta: dict with video name, frame indices, phases

    Parameters
    ----------
    sequences_dir : str or Path
        Directory containing VIDxx_sequence.json files.
    video_ids : list of str
        Which videos to include (e.g. ['VID01', 'VID05']).
    window_size : int
        Number of consecutive frames to use as input context.
    stride : int
        Step size between consecutive samples within a video.
    """

    def __init__(
        self,
        sequences_dir: str | Path,
        video_ids: list[str],
        window_size: int = 10,
        stride: int = 1,
    ):
        self.sequences_dir = Path(sequences_dir)
        self.window_size = window_size
        self.stride = stride

        # Load label mapping
        mapping_path = self.sequences_dir / "triplet_label_mapping.json"
        with open(mapping_path) as f:
            self.label_mapping = json.load(f)
        self.num_classes = len(self.label_mapping)

        # Load and index all samples
        self.samples = []  # list of (video_id, start_idx) tuples
        self.video_data = {}  # video_id -> list of multi-hot arrays

        for vid_id in video_ids:
            seq_path = self.sequences_dir / f"{vid_id}_sequence.json"
            if not seq_path.exists():
                print(f"  WARNING: {seq_path} not found, skipping.")
                continue

            with open(seq_path) as f:
                seq = json.load(f)

            # Pre-compute multi-hot arrays for efficiency
            multi_hots = np.array(
                [frame["multi_hot"] for frame in seq], dtype=np.float32
            )
            phases = [frame["phase"] for frame in seq]
            frame_ids = [frame["frame"] for frame in seq]

            self.video_data[vid_id] = {
                "multi_hots": multi_hots,
                "phases": phases,
                "frame_ids": frame_ids,
            }

            # Create sample indices with sliding window
            n_frames = len(seq)
            for start in range(0, n_frames - window_size, stride):
                self.samples.append((vid_id, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, start = self.samples[idx]
        data = self.video_data[vid_id]

        # Input: window of multi-hot frames
        end = start + self.window_size
        x = torch.from_numpy(data["multi_hots"][start:end])  # (window, C)

        # Target: next frame after the window
        y = torch.from_numpy(data["multi_hots"][end])  # (C,)

        return x, y

    def get_meta(self, idx):
        """Get metadata for a sample (not used in training, useful for analysis)."""
        vid_id, start = self.samples[idx]
        data = self.video_data[vid_id]
        end = start + self.window_size
        return {
            "video": vid_id,
            "input_frames": data["frame_ids"][start:end],
            "target_frame": data["frame_ids"][end],
            "input_phases": data["phases"][start:end],
            "target_phase": data["phases"][end],
        }


def build_dataloaders(
    sequences_dir: str | Path,
    splits_path: str | Path,
    window_size: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
):
    """Build train/val/test DataLoaders from saved splits.

    Returns
    -------
    dict
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    int
        Number of classes
    """
    from torch.utils.data import DataLoader

    with open(splits_path) as f:
        splits = json.load(f)

    sequences_dir = Path(sequences_dir)
    loaders = {}

    for split_name, video_ids in splits.items():
        ds = TripletSequenceDataset(
            sequences_dir=sequences_dir,
            video_ids=video_ids,
            window_size=window_size,
            stride=1 if split_name == "train" else window_size,
        )
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
        print(f"  {split_name}: {len(ds)} samples from {len(video_ids)} videos")

    num_classes = ds.num_classes
    return loaders, num_classes
