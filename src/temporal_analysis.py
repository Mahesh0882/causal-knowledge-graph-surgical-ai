"""
Temporal and Co-occurrence Analysis
===================================

Implements advanced causal analyses:
1. Co-occurring Triplets (Simultaneous Actions)
2. Triplet Sequence Patterns (Transition Matrix for frame N -> N+1)
"""

import pandas as pd
import numpy as np
from itertools import combinations, product

def build_cooccurrence_matrix(df_video: pd.DataFrame) -> pd.DataFrame:
    """Build a co-occurrence matrix of triplets.

    Finds how often Triplet A and Triplet B appear in the exact same frame.

    Parameters
    ----------
    df_video : pd.DataFrame
        Parsed valid triplets (typically for a single video).

    Returns
    -------
    pd.DataFrame
        Symmetric co-occurrence matrix where index and columns are triplet labels.
    """
    # Only need to check frames with multiple labels
    df_multi = df_video[df_video['is_multi_label']].copy()
    if df_multi.empty:
        return pd.DataFrame()

    cooccurrences = []

    # Group by frame to see which triplets are active simultaneously
    for frame_id, group in df_multi.groupby('frame'):
        triplets = group['triplet_label'].unique().tolist()
        if len(triplets) > 1:
            # Add all pairs 
            for t1, t2 in combinations(sorted(triplets), 2):
                cooccurrences.append((t1, t2))
                cooccurrences.append((t2, t1))  # Make symmetric
                
    if not cooccurrences:
        return pd.DataFrame()

    df_pairs = pd.DataFrame(cooccurrences, columns=['Triplet_A', 'Triplet_B'])
    
    # Calculate counts
    matrix = pd.crosstab(df_pairs['Triplet_A'], df_pairs['Triplet_B'])
    
    return matrix


def build_transition_matrix(df_video: pd.DataFrame, max_gap: int = 1) -> pd.DataFrame:
    """Build a matrix mapping Triplet A -> Triplet B in chronological order.

    If Triplet A is active at Frame N, and Triplet B is active at Frame N+1
    (or up to max_gap), we record A -> B. This establishes a causal sequence.

    Parameters
    ----------
    df_video : pd.DataFrame
        Parsed valid triplets.
    max_gap : int
        Maximum frames between N and N+gap to count as a transition. 
        Usually 1 means strictly the very next frame.

    Returns
    -------
    pd.DataFrame
        Transition matrix (rows=source, cols=target).
    """
    # Create an easy lookup for triplets active per frame
    active_per_frame = (
        df_video.groupby('frame')['triplet_label']
        .apply(lambda x: list(set(x)))
        .to_dict()
    )
    
    frames = sorted(active_per_frame.keys())
    transitions = []

    for i, frame in enumerate(frames[:-1]):
        current_triplets = active_per_frame[frame]
        
        # Check upcoming frames within the gap
        # Since we use frame IDs, we strictly check for actual frame integers N+1
        next_frames = [f for f in frames[i+1:] if f - frame <= max_gap]
        
        for next_f in next_frames:
            next_triplets = active_per_frame[next_f]
            # Record transitions from every current triplet to every next triplet
            for t_curr, t_next in product(current_triplets, next_triplets):
                transitions.append((t_curr, t_next))

    if not transitions:
        return pd.DataFrame()

    df_trans = pd.DataFrame(transitions, columns=['From_Triplet', 'To_Triplet'])
    matrix = pd.crosstab(df_trans['From_Triplet'], df_trans['To_Triplet'])
    
    return matrix
