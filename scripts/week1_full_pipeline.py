"""
Week 1: Full Dataset Pipeline
==============================
Parses all CholecT50 videos, builds graphs, generates summary stats,
and creates the temporal graph sequence dataset for downstream modeling.

Usage:
    python scripts/week1_full_pipeline.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import networkx as nx

from src.triplet_parser import (
    parse_video,
    filter_valid_triplets,
    video_summary,
    save_parsed_csv,
    load_categories,
    multi_label_analysis,
)
from src.graph_builder import build_graph, save_graph, graph_stats


# ---- Configuration ----
LABELS_DIR = Path("/Users/maheshkundurthi/Documents/Research work OPT/CholecT50/labels")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "parsed_triplets"
GRAPH_DIR = OUTPUT_DIR / "graphs"
STATS_DIR = OUTPUT_DIR / "stats"


def parse_all_videos():
    """Step 1: Parse all CholecT50 video JSONs into CSVs."""
    print("\n" + "=" * 60)
    print("STEP 1: Parsing all CholecT50 video JSONs")
    print("=" * 60)

    json_files = sorted(LABELS_DIR.glob("VID*.json"))
    print(f"Found {len(json_files)} video JSONs in {LABELS_DIR}\n")

    all_dfs = []
    results = []

    for json_path in json_files:
        vid_name = json_path.stem
        try:
            t0 = time.time()
            df = parse_video(json_path)
            df_valid = filter_valid_triplets(df)
            elapsed = time.time() - t0

            # Save per-video CSV
            csv_path = CSV_DIR / f"{vid_name}_triplets.csv"
            df_valid.to_csv(csv_path, index=False)

            all_dfs.append(df_valid)
            results.append({
                "video": vid_name,
                "status": "OK",
                "total_rows": len(df),
                "valid_rows": len(df_valid),
                "frames": df_valid["frame"].nunique(),
                "unique_triplets": df_valid["triplet_id"].nunique(),
                "parse_time_sec": round(elapsed, 2),
            })
            print(f"  [OK] {vid_name}: {len(df_valid)} valid rows, "
                  f"{df_valid['frame'].nunique()} frames ({elapsed:.1f}s)")

        except Exception as e:
            results.append({
                "video": vid_name,
                "status": f"FAILED: {e}",
                "total_rows": 0,
                "valid_rows": 0,
                "frames": 0,
                "unique_triplets": 0,
                "parse_time_sec": 0,
            })
            print(f"  [FAIL] {vid_name}: {e}")

    # Save combined CSV
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        all_csv = CSV_DIR / "all_triplets.csv"
        df_all.to_csv(all_csv, index=False)
        print(f"\nSaved combined CSV: {all_csv} ({len(df_all)} rows)")

    # Save parse report
    df_report = pd.DataFrame(results)
    report_path = STATS_DIR / "parse_report.csv"
    df_report.to_csv(report_path, index=False)
    print(f"Saved parse report: {report_path}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(results) - ok_count
    print(f"\nResults: {ok_count} succeeded, {fail_count} failed out of {len(results)}")

    return df_all if all_dfs else pd.DataFrame()


def build_all_graphs(df_all: pd.DataFrame):
    """Step 2: Build NetworkX graphs for all videos."""
    print("\n" + "=" * 60)
    print("STEP 2: Building graphs for all videos")
    print("=" * 60)

    all_stats = []

    for vid_name, df_vid in df_all.groupby("video"):
        try:
            G = build_graph(df_vid, video_name=vid_name)
            save_graph(G, GRAPH_DIR / f"{vid_name}_graph.gexf")

            stats = graph_stats(G)
            all_stats.append(stats)
            print(f"  [OK] {vid_name}: {stats['nodes']} nodes, {stats['edges']} edges")

        except Exception as e:
            print(f"  [FAIL] {vid_name}: {e}")

    # Save graph stats summary
    df_stats = pd.DataFrame(all_stats)
    stats_path = STATS_DIR / "graph_stats_all.csv"
    df_stats.to_csv(stats_path, index=False)
    print(f"\nSaved graph stats: {stats_path}")

    return df_stats


def generate_dataset_summary(df_all: pd.DataFrame):
    """Step 3: Generate comprehensive dataset statistics."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating dataset summary statistics")
    print("=" * 60)

    # Per-video summary
    summary = video_summary(df_all)
    summary_path = STATS_DIR / "video_summary_all.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved video summary: {summary_path}")

    # Global statistics
    global_stats = {
        "total_videos": df_all["video"].nunique(),
        "total_frames": df_all["frame"].nunique(),
        "total_triplet_rows": len(df_all),
        "unique_triplet_classes": df_all["triplet_id"].nunique(),
        "unique_instruments": df_all["instrument"].nunique(),
        "unique_verbs": df_all["verb"].nunique(),
        "unique_targets": df_all["target"].nunique(),
        "instruments": sorted(df_all["instrument"].unique().tolist()),
        "verbs": sorted(df_all["verb"].unique().tolist()),
        "targets": sorted(df_all["target"].unique().tolist()),
    }

    # Multi-label analysis
    ml_stats = multi_label_analysis(df_all)
    global_stats.update({
        "multi_label_frames": ml_stats["multi_label_frames"],
        "multi_label_pct": ml_stats["multi_label_pct"],
        "max_triplets_per_frame": ml_stats["max_triplets_per_frame"],
        "mean_triplets_per_frame": ml_stats["mean_triplets_per_frame"],
    })

    # Save as JSON
    stats_json_path = STATS_DIR / "global_dataset_stats.json"
    with open(stats_json_path, "w") as f:
        json.dump(global_stats, f, indent=2)
    print(f"Saved global stats: {stats_json_path}")

    # Print summary
    print(f"\n--- Global Dataset Summary ---")
    print(f"  Videos:           {global_stats['total_videos']}")
    print(f"  Total frames:     {global_stats['total_frames']}")
    print(f"  Total rows:       {global_stats['total_triplet_rows']}")
    print(f"  Triplet classes:  {global_stats['unique_triplet_classes']}")
    print(f"  Instruments:      {global_stats['unique_instruments']}")
    print(f"  Verbs:            {global_stats['unique_verbs']}")
    print(f"  Targets:          {global_stats['unique_targets']}")
    print(f"  Multi-label:      {global_stats['multi_label_pct']}% of frames")
    print(f"  Max per frame:    {global_stats['max_triplets_per_frame']}")

    # Triplet class distribution
    triplet_dist = df_all.groupby("triplet_label").size().sort_values(ascending=False)
    triplet_dist_path = STATS_DIR / "triplet_class_distribution.csv"
    triplet_dist.to_csv(triplet_dist_path, header=["count"])
    print(f"Saved triplet distribution: {triplet_dist_path}")

    # Phase distribution
    if "phase" in df_all.columns:
        phase_dist = df_all.groupby("phase").agg(
            frames=("frame", "nunique"),
            triplet_rows=("triplet_id", "count"),
        ).sort_values("frames", ascending=False)
        phase_path = STATS_DIR / "phase_distribution.csv"
        phase_dist.to_csv(phase_path)
        print(f"Saved phase distribution: {phase_path}")

    return global_stats


def build_temporal_sequences(df_all: pd.DataFrame):
    """Step 4: Build temporal graph sequences for model training.

    For each video, create a chronological sequence of graph states.
    Each state captures the active triplets at that frame.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Building temporal graph sequences")
    print("=" * 60)

    sequences_dir = OUTPUT_DIR / "temporal_sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    # Get the full set of triplet classes for consistent encoding
    all_triplet_labels = sorted(df_all["triplet_label"].unique().tolist())
    triplet_to_idx = {label: idx for idx, label in enumerate(all_triplet_labels)}
    num_classes = len(all_triplet_labels)

    # Save the label mapping
    mapping_path = sequences_dir / "triplet_label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(triplet_to_idx, f, indent=2)
    print(f"Saved label mapping: {mapping_path} ({num_classes} classes)")

    # Also save instrument/verb/target mappings
    instrument_labels = sorted(df_all["instrument"].unique().tolist())
    verb_labels = sorted(df_all["verb"].unique().tolist())
    target_labels = sorted(df_all["target"].unique().tolist())

    entity_mapping = {
        "instruments": {label: idx for idx, label in enumerate(instrument_labels)},
        "verbs": {label: idx for idx, label in enumerate(verb_labels)},
        "targets": {label: idx for idx, label in enumerate(target_labels)},
    }
    entity_path = sequences_dir / "entity_label_mapping.json"
    with open(entity_path, "w") as f:
        json.dump(entity_mapping, f, indent=2)
    print(f"Saved entity mapping: {entity_path}")

    for vid_name, df_vid in df_all.groupby("video"):
        frames = sorted(df_vid["frame"].unique())

        sequence_data = []
        for frame_id in frames:
            frame_df = df_vid[df_vid["frame"] == frame_id]

            # Multi-hot encoding over triplet classes
            multi_hot = np.zeros(num_classes, dtype=np.int8)
            active_triplet_indices = []
            for label in frame_df["triplet_label"].unique():
                if label in triplet_to_idx:
                    idx = triplet_to_idx[label]
                    multi_hot[idx] = 1
                    active_triplet_indices.append(idx)

            # Structured graph state for this frame
            active_edges = []
            for _, row in frame_df.iterrows():
                inst_idx = entity_mapping["instruments"].get(row["instrument"], -1)
                verb_idx = entity_mapping["verbs"].get(row["verb"], -1)
                tgt_idx = entity_mapping["targets"].get(row["target"], -1)
                active_edges.append({
                    "instrument": row["instrument"],
                    "instrument_idx": inst_idx,
                    "verb": row["verb"],
                    "verb_idx": verb_idx,
                    "target": row["target"],
                    "target_idx": tgt_idx,
                    "triplet_label": row["triplet_label"],
                    "triplet_idx": triplet_to_idx.get(row["triplet_label"], -1),
                })

            phase = frame_df["phase"].iloc[0] if "phase" in frame_df.columns else "unknown"

            sequence_data.append({
                "frame": int(frame_id),
                "phase": phase,
                "num_active_triplets": len(active_triplet_indices),
                "active_triplet_indices": active_triplet_indices,
                "multi_hot": multi_hot.tolist(),
                "edges": active_edges,
            })

        # Save as JSON
        seq_path = sequences_dir / f"{vid_name}_sequence.json"
        with open(seq_path, "w") as f:
            json.dump(sequence_data, f)
        print(f"  [OK] {vid_name}: {len(sequence_data)} frames")

    print(f"\nAll sequences saved to {sequences_dir}")


def create_data_splits(df_all: pd.DataFrame):
    """Step 5: Create train/val/test splits.

    Uses a fixed split for reproducibility.
    CholecT45 official splits are used when available.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Creating train/val/test splits")
    print("=" * 60)

    all_videos = sorted(df_all["video"].unique().tolist())
    n = len(all_videos)
    print(f"Total videos available: {n}")

    # Fixed split (reproducible)
    # CholecT45 commonly uses a ~60/10/30 or 70/15/15 split
    # We use roughly 70% train / 15% val / 15% test
    np.random.seed(42)
    shuffled = np.random.permutation(all_videos).tolist()

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_vids = sorted(shuffled[:n_train])
    val_vids = sorted(shuffled[n_train:n_train + n_val])
    test_vids = sorted(shuffled[n_train + n_val:])

    splits = {
        "train": train_vids,
        "val": val_vids,
        "test": test_vids,
    }

    splits_path = OUTPUT_DIR / "data_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"  Train: {len(train_vids)} videos -> {train_vids}")
    print(f"  Val:   {len(val_vids)} videos  -> {val_vids}")
    print(f"  Test:  {len(test_vids)} videos  -> {test_vids}")
    print(f"Saved splits: {splits_path}")

    return splits


# ---- Main ----
if __name__ == "__main__":
    # Create output directories
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Step 1: Parse all videos
    df_all = parse_all_videos()

    if df_all.empty:
        print("\nERROR: No videos were parsed successfully. Exiting.")
        sys.exit(1)

    # Step 2: Build graphs
    graph_stats_df = build_all_graphs(df_all)

    # Step 3: Dataset summary
    global_stats = generate_dataset_summary(df_all)

    # Step 4: Temporal sequences
    build_temporal_sequences(df_all)

    # Step 5: Data splits
    splits = create_data_splits(df_all)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"WEEK 1 PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 60)
    print(f"  Videos parsed:      {global_stats['total_videos']}")
    print(f"  Total frames:       {global_stats['total_frames']}")
    print(f"  Triplet classes:    {global_stats['unique_triplet_classes']}")
    print(f"  Output directory:   {OUTPUT_DIR}")
