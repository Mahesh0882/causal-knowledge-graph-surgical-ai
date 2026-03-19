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

    return {
        'triplet_id': triplet_id,
        'triplet_label': triplet_label,
        'instrument_id': instrument_id,
        'instrument': instrument,
        'verb_id': verb_id,
        'verb': verb,
        'target_id': target_id,
        'target': target,
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
        for entry in entries:
            decoded = decode_annotation_entry(entry, categories)
            decoded['frame'] = int(frame_id)
            decoded['video'] = video_name
            decoded['num_triplets_in_frame'] = len(entries)
            rows.append(decoded)

    df = pd.DataFrame(rows)
    df = df.sort_values('frame').reset_index(drop=True)

    col_order = [
        'video', 'frame', 'num_triplets_in_frame',
        'triplet_id', 'triplet_label',
        'instrument_id', 'instrument',
        'verb_id', 'verb',
        'target_id', 'target',
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
# CLI Entry Point (optional quick test)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python triplet_parser.py <path_to_video.json>')
        sys.exit(1)

    path = sys.argv[1]
    df = parse_video(path)
    print(f'\nParsed {path}:')
    print(f'  Total rows: {len(df)}')
    print(f'  Frames: {df["frame"].nunique()}')
    print(f'\nFirst 10 rows:')
    print(df.head(10).to_string(index=False))
