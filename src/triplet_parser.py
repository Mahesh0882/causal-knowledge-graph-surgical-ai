"""
Triplet Parser for CholecT50 Surgical Action Annotations
=========================================================
Loads CholecT50 JSON annotation files and converts binary annotation
arrays into human-readable (instrument, verb, target) triplets.

Handles multi-label frames where multiple triplets are active simultaneously.

Usage:
    from src.triplet_parser import load_categories, parse_video, parse_multiple_videos
"""

from __future__ import annotations

import json
import os
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Category / Dictionary Loading
# ---------------------------------------------------------------------------

def load_categories(json_path: str | Path) -> dict:
    """Load the category dictionaries from any CholecT50 video JSON.

    The categories (instrument, verb, target, triplet name mappings)
    are embedded in every video JSON under the 'categories' key.

    Parameters
    ----------
    json_path : str or Path
        Path to any CholecT50 video annotation JSON file.

    Returns
    -------
    dict
        Dictionary with keys 'instrument', 'verb', 'target', 'triplet',
        each mapping string IDs to label names.
    """
    with open(json_path) as f:
        data = json.load(f)
    return data['categories']


def print_categories(categories: dict) -> None:
    """Pretty-print all category dictionaries for inspection."""
    sections = [
        ('INSTRUMENTS', 'instrument'),
        ('VERBS', 'verb'),
        ('TARGETS', 'target'),
    ]
    for title, key in sections:
        print(f'\n=== {title} ({len(categories[key])} classes) ===')
        for k, v in categories[key].items():
            print(f'  {k}: {v}')

    print(f'\n=== TRIPLETS ({len(categories["triplet"])} classes) ===')
    for k, v in list(categories['triplet'].items())[:10]:
        print(f'  {k}: {v}')
    print(f'  ... ({len(categories["triplet"])} total)')


# ---------------------------------------------------------------------------
# Single-Entry Decoding
# ---------------------------------------------------------------------------

def decode_annotation_entry(entry: list, categories: dict) -> dict:
    """Decode a single 15-element CholecT50 annotation entry.

    Annotation format (per entry):
        [triplet_id, instrument_id, instrument_presence,
         bbox_x, bbox_y, bbox_w, bbox_h,
         verb_id, target_id, ivt_presence,
         bbox_x, bbox_y, bbox_w, bbox_h,
         phase_annotation]

    Parameters
    ----------
    entry : list
        15-element annotation array from the JSON.
    categories : dict
        Category dictionaries from load_categories().

    Returns
    -------
    dict
        Decoded triplet with keys: triplet_id, triplet_label,
        instrument_id, instrument, verb_id, verb, target_id, target.
    """
    triplet_id = entry[0]
    instrument_id = entry[1]
    verb_id = entry[7]
    target_id = entry[8]

    instrument = categories['instrument'].get(
        str(instrument_id), f'unknown_{instrument_id}'
    )
    verb = categories['verb'].get(
        str(verb_id), f'unknown_{verb_id}'
    )
    target = categories['target'].get(
        str(target_id), f'unknown_{target_id}'
    )
    triplet_label = categories['triplet'].get(
        str(triplet_id), f'unknown_{triplet_id}'
    )

    # Flag invalid / null annotations (ID == -1)
    is_valid = triplet_id >= 0

    # Flag null_verb / null_target entries
    is_null = (verb_id == 9) or (target_id == 14)  # null_verb=9, null_target=14

    return {
        'triplet_id': triplet_id,
        'triplet_label': triplet_label,
        'instrument_id': instrument_id,
        'instrument': instrument,
        'verb_id': verb_id,
        'verb': verb,
        'target_id': target_id,
        'target': target,
        'is_valid': is_valid,
        'is_null': is_null,
    }


# ---------------------------------------------------------------------------
# Video-Level Parsing
# ---------------------------------------------------------------------------

def parse_video(json_path: str | Path) -> pd.DataFrame:
    """Parse all frames of a CholecT50 video JSON into a tidy DataFrame.

    Each active triplet in each frame becomes its own row.
    Multi-label frames (where multiple triplets are simultaneously active)
    are fully expanded — the column ``num_triplets_in_frame`` records how
    many triplets were active in that frame.

    Parameters
    ----------
    json_path : str or Path
        Path to a CholecT50 video annotation JSON (e.g. VID01.json).

    Returns
    -------
    pd.DataFrame
        Columns: video, frame, num_triplets_in_frame, triplet_id,
        triplet_label, instrument_id, instrument, verb_id, verb,
        target_id, target.
    """
    with open(json_path) as f:
        data = json.load(f)

    categories = data['categories']
    annotations = data['annotations']
    video_name = data.get('video', Path(json_path).stem)

    rows = []
    for frame_id, entries in annotations.items():
        for entry_idx, entry in enumerate(entries):
            decoded = decode_annotation_entry(entry, categories)
            decoded['frame'] = int(frame_id)
            decoded['video'] = video_name
            decoded['num_triplets_in_frame'] = len(entries)
            decoded['entry_index'] = entry_idx  # preserve raw order
            rows.append(decoded)

    df = pd.DataFrame(rows)
    # Stable sort: preserves annotation order within each frame
    df = df.sort_values(['frame', 'entry_index']).reset_index(drop=True)

    # Mark multi-label frames
    df['is_multi_label'] = df['num_triplets_in_frame'] > 1

    col_order = [
        'video', 'frame', 'entry_index',
        'num_triplets_in_frame', 'is_multi_label',
        'triplet_id', 'triplet_label',
        'instrument_id', 'instrument',
        'verb_id', 'verb',
        'target_id', 'target',
        'is_valid', 'is_null',
    ]
    return df[col_order]


def parse_multiple_videos(triplet_dir: str | Path,
                          video_ids: list[str] | None = None) -> pd.DataFrame:
    """Parse multiple video JSONs and concatenate into one DataFrame.

    Parameters
    ----------
    triplet_dir : str or Path
        Directory containing video annotation JSON files.
    video_ids : list of str, optional
        Specific video IDs to parse (e.g. ['VID01', 'VID05', 'VID40']).
        If None, all JSON files in the directory are parsed.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all parsed triplets.
    """
    triplet_dir = Path(triplet_dir)

    if video_ids is not None:
        json_files = [triplet_dir / f'{vid}.json' for vid in video_ids]
    else:
        json_files = sorted(triplet_dir.glob('*.json'))

    dfs = []
    for json_file in json_files:
        if not json_file.exists():
            print(f'  WARNING: {json_file} not found, skipping.')
            continue
        df = parse_video(json_file)
        print(f'  Parsed {json_file.stem}: {len(df)} rows, '
              f'{df["frame"].nunique()} frames')
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f'No valid JSON files found in {triplet_dir}')

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary / Statistics
# ---------------------------------------------------------------------------

def video_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-video summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Output of parse_video() or parse_multiple_videos().

    Returns
    -------
    pd.DataFrame
        Summary with total frames, total triplet rows, unique triplets,
        multi-label frame count, and mean triplets per frame.
    """
    summaries = []
    for video, vdf in df.groupby('video'):
        n_frames = vdf['frame'].nunique()
        multi = (vdf.groupby('frame').size() > 1).sum()
        summaries.append({
            'video': video,
            'total_frames': n_frames,
            'total_triplet_rows': len(vdf),
            'unique_triplets': vdf['triplet_id'].nunique(),
            'multi_label_frames': int(multi),
            'multi_label_pct': round(multi / n_frames * 100, 1),
            'mean_triplets_per_frame': round(len(vdf) / n_frames, 2),
        })
    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Multi-Label Frame Utilities
# ---------------------------------------------------------------------------

def filter_valid_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where triplet_id == -1 (no annotation / null).

    After filtering, ``num_triplets_in_frame`` and ``is_multi_label``
    are recalculated to reflect only valid entries.

    Parameters
    ----------
    df : pd.DataFrame
        Output of parse_video() or parse_multiple_videos().

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only valid triplets.
    """
    df = df[df['is_valid']].copy()
    # Recalculate per-frame counts after filtering
    frame_counts = df.groupby(['video', 'frame']).size().rename('num_triplets_in_frame')
    df = df.drop(columns=['num_triplets_in_frame']).merge(
        frame_counts, on=['video', 'frame'], how='left'
    )
    df['is_multi_label'] = df['num_triplets_in_frame'] > 1
    return df.reset_index(drop=True)


def multi_label_analysis(df: pd.DataFrame) -> dict:
    """Detailed analysis of multi-label frames in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Output of parse_video() or parse_multiple_videos().

    Returns
    -------
    dict
        Dictionary with multi-label statistics.
    """
    frame_counts = df.groupby(['video', 'frame']).size()
    n_frames = len(frame_counts)
    multi = (frame_counts > 1).sum()

    return {
        'total_frames': n_frames,
        'single_label_frames': int((frame_counts == 1).sum()),
        'multi_label_frames': int(multi),
        'multi_label_pct': round(multi / n_frames * 100, 1) if n_frames else 0,
        'max_triplets_per_frame': int(frame_counts.max()) if n_frames else 0,
        'mean_triplets_per_frame': round(frame_counts.mean(), 2) if n_frames else 0,
        'distribution': frame_counts.value_counts().sort_index().to_dict(),
    }


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def save_parsed_csv(df: pd.DataFrame, output_dir: str | Path,
                    per_video: bool = True) -> list[Path]:
    """Save parsed triplet DataFrame(s) to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of parse_video() or parse_multiple_videos().
    output_dir : str or Path
        Directory to write CSV files to.
    per_video : bool
        If True, save one CSV per video. If False, save a single combined CSV.

    Returns
    -------
    list of Path
        Paths to the saved CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if per_video:
        for video, vdf in df.groupby('video'):
            fname = output_dir / f'{video}_triplets.csv'
            vdf.to_csv(fname, index=False)
            print(f'  Saved {fname.name} ({len(vdf)} rows)')
            saved.append(fname)
    else:
        fname = output_dir / 'all_triplets.csv'
        df.to_csv(fname, index=False)
        print(f'  Saved {fname.name} ({len(df)} rows)')
        saved.append(fname)

    return saved


# ---------------------------------------------------------------------------
# CLI Entry Point (optional quick test)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python triplet_parser.py <path_to_video.json>')
        sys.exit(1)

    path = sys.argv[1]
    df = parse_video(path)
    df_valid = filter_valid_triplets(df)

    print(f'\nParsed {path}:')
    print(f'  Total rows (raw):   {len(df)}')
    print(f'  Valid rows:         {len(df_valid)}')
    print(f'  Frames:             {df_valid["frame"].nunique()}')

    stats = multi_label_analysis(df_valid)
    print(f'\n  Multi-label frames: {stats["multi_label_frames"]} '
          f'({stats["multi_label_pct"]}%)')
    print(f'  Max per frame:      {stats["max_triplets_per_frame"]}')
    print(f'  Distribution:       {stats["distribution"]}')

    print(f'\nFirst 10 valid rows:')
    print(df_valid.head(10).to_string(index=False))
