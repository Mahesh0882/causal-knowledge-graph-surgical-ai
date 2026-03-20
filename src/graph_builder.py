"""
Graph Builder for CholecT50 Surgical Action Triplets
=====================================================
Constructs NetworkX MultiDiGraph objects from parsed triplet CSVs.

Graph schema:
  - Nodes: instruments (node_type='instrument') + targets (node_type='target')
  - Edges: one per unique (instrument, verb, target) triplet
  - Edge key: verb name
  - Edge attrs: frequency, frame_list, first/last_frame, timestamps

Usage:
    from src.graph_builder import build_graph, build_graphs_for_videos, graph_stats
"""

from __future__ import annotations

import json
import pandas as pd
import networkx as nx
from pathlib import Path


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph(df: pd.DataFrame, video_name: str | None = None) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from a parsed triplet DataFrame.

    Filters out null_verb/null_target rows (is_null=True) and invalid
    rows (is_valid=False) before constructing the graph.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed triplet DataFrame (output of triplet_parser.parse_video
        or loaded from a CSV).
    video_name : str, optional
        Video identifier. If None, inferred from the 'video' column.

    Returns
    -------
    nx.MultiDiGraph
        Directed multigraph with instrument→target edges keyed by verb.
    """
    # Infer video name if not provided
    if video_name is None:
        video_name = str(df['video'].iloc[0]) if 'video' in df.columns else 'unknown'

    # Filter to valid, non-null triplets only
    mask = True
    if 'is_valid' in df.columns:
        mask = mask & df['is_valid']
    if 'is_null' in df.columns:
        mask = mask & ~df['is_null']
    df_clean = df[mask].copy()

    # --- Aggregate edges ---
    # Each unique (instrument, verb, target) becomes one edge
    edge_agg = (
        df_clean.groupby(['instrument', 'verb', 'target', 'triplet_id', 'triplet_label'])
        .agg(
            frequency=('frame', 'nunique'),
            frame_list=('frame', lambda x: sorted(x.unique().tolist())),
            first_frame=('frame', 'min'),
            last_frame=('frame', 'max'),
        )
        .reset_index()
    )

    # Add timestamp columns if available
    if 'timestamp_sec' in df_clean.columns:
        ts_agg = (
            df_clean.groupby(['instrument', 'verb', 'target'])
            .agg(
                timestamp_start=('timestamp_sec', 'min'),
                timestamp_end=('timestamp_sec', 'max'),
            )
            .reset_index()
        )
        edge_agg = edge_agg.merge(ts_agg, on=['instrument', 'verb', 'target'], how='left')

    # Compute duration
    edge_agg['duration_frames'] = edge_agg['last_frame'] - edge_agg['first_frame'] + 1

    # --- Build the graph ---
    G = nx.MultiDiGraph()

    # Graph-level metadata
    G.graph['video'] = video_name
    G.graph['total_frames'] = int(df_clean['frame'].nunique())
    G.graph['total_triplet_rows'] = len(df_clean)
    G.graph['unique_triplets'] = int(df_clean['triplet_id'].nunique())

    # Add instrument nodes
    for inst in df_clean['instrument'].unique():
        inst_id = int(df_clean[df_clean['instrument'] == inst]['instrument_id'].iloc[0])
        G.add_node(inst, node_type='instrument', id=inst_id)

    # Add target nodes
    for tgt in df_clean['target'].unique():
        tgt_id = int(df_clean[df_clean['target'] == tgt]['target_id'].iloc[0])
        G.add_node(tgt, node_type='target', id=tgt_id)

    # Add edges
    for _, row in edge_agg.iterrows():
        attrs = {
            'verb': row['verb'],
            'triplet_id': int(row['triplet_id']),
            'triplet_label': row['triplet_label'],
            'frequency': int(row['frequency']),
            'frame_list': row['frame_list'],
            'first_frame': int(row['first_frame']),
            'last_frame': int(row['last_frame']),
            'duration_frames': int(row['duration_frames']),
        }
        if 'timestamp_start' in row:
            attrs['timestamp_start'] = float(row['timestamp_start'])
            attrs['timestamp_end'] = float(row['timestamp_end'])

        G.add_edge(row['instrument'], row['target'], key=row['verb'], **attrs)

    return G


# ---------------------------------------------------------------------------
# Multi-Video Construction
# ---------------------------------------------------------------------------

def build_graphs_for_videos(
    csv_dir: str | Path,
    video_ids: list[str] | None = None,
) -> dict[str, nx.MultiDiGraph]:
    """Build graphs for multiple videos from their parsed CSV files.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing VIDxx_triplets.csv files.
    video_ids : list of str, optional
        Specific video IDs (e.g. ['VID01', 'VID05']). If None, all CSVs
        in the directory are processed.

    Returns
    -------
    dict
        Mapping of video_id -> MultiDiGraph.
    """
    csv_dir = Path(csv_dir)

    if video_ids is not None:
        csv_files = [(vid, csv_dir / f'{vid}_triplets.csv') for vid in video_ids]
    else:
        csv_files = [
            (f.stem.replace('_triplets', ''), f)
            for f in sorted(csv_dir.glob('*_triplets.csv'))
            if f.stem != 'all_triplets'
        ]

    graphs = {}
    for vid, csv_path in csv_files:
        if not csv_path.exists():
            print(f'  WARNING: {csv_path} not found, skipping.')
            continue
        df = pd.read_csv(csv_path)
        G = build_graph(df, video_name=vid)
        print(f'  {vid}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
        graphs[vid] = G

    return graphs


# ---------------------------------------------------------------------------
# Graph Statistics
# ---------------------------------------------------------------------------

def graph_stats(G: nx.MultiDiGraph) -> dict:
    """Compute summary statistics for a surgical action graph.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph built by build_graph().

    Returns
    -------
    dict
        Statistics dictionary.
    """
    # Separate node types
    instruments = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'instrument']
    targets = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']

    # Find most frequent triplet
    most_freq_edge = None
    max_freq = 0
    for u, v, k, d in G.edges(data=True, keys=True):
        if d.get('frequency', 0) > max_freq:
            max_freq = d['frequency']
            most_freq_edge = (u, k, v, max_freq)

    # Graph density: edges / (n_instruments * n_targets)
    max_possible = len(instruments) * len(targets) if instruments and targets else 1
    density = G.number_of_edges() / max_possible

    return {
        'video': G.graph.get('video', 'unknown'),
        'nodes': G.number_of_nodes(),
        'n_instruments': len(instruments),
        'n_targets': len(targets),
        'edges': G.number_of_edges(),
        'density': round(density, 2),
        'total_frames': G.graph.get('total_frames', 0),
        'unique_triplets': G.graph.get('unique_triplets', 0),
        'most_frequent_triplet': most_freq_edge,
        'instruments': sorted(instruments),
        'targets': sorted(targets),
    }


def print_graph_stats(G: nx.MultiDiGraph) -> None:
    """Pretty-print graph statistics."""
    s = graph_stats(G)
    print(f"\n=== {s['video']} Graph Stats ===")
    print(f"Nodes: {s['nodes']} ({s['n_instruments']} instruments + {s['n_targets']} targets)")
    print(f"Edges: {s['edges']}")
    if s['most_frequent_triplet']:
        u, verb, v, freq = s['most_frequent_triplet']
        print(f"Most frequent triplet: ({u}, {verb}, {v}) - {freq} frames")
    print(f"Graph density: {s['density']}")
    print(f"Total frames: {s['total_frames']}")
    print(f"Instruments: {s['instruments']}")
    print(f"Targets: {s['targets']}")


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def save_graph(G: nx.MultiDiGraph, output_path: str | Path) -> Path:
    """Save graph to GEXF format.

    Note: frame_list (list attribute) is converted to a JSON string
    for GEXF compatibility.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph to save.
    output_path : str or Path
        Output file path (should end in .gexf).

    Returns
    -------
    Path
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GEXF doesn't support list attributes — convert to JSON strings
    G_copy = G.copy()
    for u, v, k, d in G_copy.edges(data=True, keys=True):
        if 'frame_list' in d:
            d['frame_list'] = json.dumps(d['frame_list'])

    nx.write_gexf(G_copy, str(output_path))
    print(f'  Saved {output_path.name}')
    return output_path


def load_graph(gexf_path: str | Path) -> nx.MultiDiGraph:
    """Load a graph from GEXF format.

    Restores frame_list from JSON string back to a Python list.

    Parameters
    ----------
    gexf_path : str or Path
        Path to a .gexf file.

    Returns
    -------
    nx.MultiDiGraph
    """
    G = nx.read_gexf(str(gexf_path))
    # Restore frame_list from JSON string
    for u, v, k, d in G.edges(data=True, keys=True):
        if 'frame_list' in d and isinstance(d['frame_list'], str):
            d['frame_list'] = json.loads(d['frame_list'])
    return G


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python graph_builder.py <path_to_triplets.csv>')
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    G = build_graph(df)
    print_graph_stats(G)
