"""
Edge Case Analysis for Surgical Action Triplets
================================================

Implements three edge case analyses:
    1. Phase Transition Analysis (subgraphs per phase)
    2. Isolated Nodes (missing dictionary elements per video)
    3. Verb Transitions (causal action chains on the same instrument-target pair)
"""

import pandas as pd
import networkx as nx
from pathlib import Path
from src.graph_builder import build_graph

from src.triplet_parser import load_categories

def get_missing_nodes(df_video: pd.DataFrame, categories: dict) -> dict:
    """Find isolated/missing instruments and targets for a single video.

    Compares the full dictionary of instruments and targets against
    what actually appears in the video's valid triplets.

    Parameters
    ----------
    df_video : pd.DataFrame
        Parsed valid triplets for a single video.
    categories : dict
        Category dictionaries loaded from the JSON via load_categories().

    Returns
    -------
    dict
        {'missing_instruments': list, 'missing_targets': list}
    """
    all_inst = set(categories.get('instrument', {}).values())
    all_tgt = set(categories.get('target', {}).values())

    used_inst = set(df_video['instrument'].unique())
    used_tgt = set(df_video['target'].unique())

    missing_inst = sorted(list(all_inst - used_inst))
    missing_tgt = sorted(list(all_tgt - used_tgt))

    return {
        'missing_instruments': missing_inst,
        'missing_targets': missing_tgt,
        'used_instruments': sorted(list(used_inst)),
        'used_targets': sorted(list(used_tgt))
    }

def build_phase_subgraphs(df_video: pd.DataFrame) -> dict[str, nx.MultiDiGraph]:
    """Build a separate graph for each surgical phase in the video.

    Parameters
    ----------
    df_video : pd.DataFrame
        Parsed valid triplets for a single video containing a 'phase' column.

    Returns
    -------
    dict
        Mapping from phase name to nx.MultiDiGraph.
    """
    subgraphs = {}
    if 'phase' not in df_video.columns:
        raise ValueError("DataFrame must contain a 'phase' column.")

    for phase_name, df_phase in df_video.groupby('phase'):
        # Pass a custom video name indicating the phase
        vid_name = str(df_video['video'].iloc[0])
        G_phase = build_graph(df_phase, video_name=f"{vid_name} - {phase_name}")
        subgraphs[phase_name] = G_phase

    return subgraphs

def find_verb_transitions(df_video: pd.DataFrame, max_frame_gap: int = 1) -> pd.DataFrame:
    """Find causal verb transitions on the same instrument-target pair.

    Looks for instances where (instrument, verb_A, target) is followed by
    (instrument, verb_B, target) in close succession (e.g. consecutive frames).

    Parameters
    ----------
    df_video : pd.DataFrame
        Parsed valid triplets for a single video.
    max_frame_gap : int
        Maximum number of frames between verb A and verb B to be considered
        a direct transition. Default is 1 (consecutive frames).

    Returns
    -------
    pd.DataFrame
        DataFrame of state transitions containing:
        instrument, target, frame_A, verb_A, frame_B, verb_B
    """
    # Sort chronologically
    df_sorted = df_video.sort_values('frame').copy()

    transitions = []

    # Group by the edges (instrument, target)
    for (inst, tgt), group in df_sorted.groupby(['instrument', 'target']):
        # We need consecutive annotations
        for i in range(len(group) - 1):
            row_a = group.iloc[i]
            row_b = group.iloc[i + 1]

            frame_gap = row_b['frame'] - row_a['frame']
            
            # Note: We only care if the verb actually changed
            if 0 < frame_gap <= max_frame_gap and row_a['verb'] != row_b['verb']:
                transitions.append({
                    'video': row_a['video'],
                    'instrument': inst,
                    'target': tgt,
                    'frame_A': row_a['frame'],
                    'verb_A': row_a['verb'],
                    'frame_B': row_b['frame'],
                    'verb_B': row_b['verb'],
                    'phase_A': row_a.get('phase', 'unknown'),
                    'phase_B': row_b.get('phase', 'unknown'),
                    'gap': frame_gap
                })

    return pd.DataFrame(transitions)
