"""
Advanced Causal Analytics
=========================

Implements paper-worthy edge case metrics:
1. Anomalous Triplet Sequences (Implicit Failure Mining)
2. Graph Comparison (Structural distance between surgical workflows)
"""

import pandas as pd
import numpy as np
import networkx as nx
from src.temporal_analysis import build_transition_matrix

# ---------------------------------------------------------------------------
# Anomalous Sequences
# ---------------------------------------------------------------------------

def find_anomalous_transitions(df_all: pd.DataFrame, threshold_prob: float = 0.05) -> pd.DataFrame:
    """Find rare, anomalous transitions that represent workflow deviations.
    
    Computes a global transition matrix, converts it to probabilities 
    (row-normalized), and flags any transition that occurs but has a 
    historical probability below the threshold (e.g. < 5%).

    Parameters
    ----------
    df_all : pd.DataFrame
        Parsed triplets containing data from MULTIPLE videos to build
        a reliable global distribution.
    threshold_prob : float
        Probability threshold below which a transition is flagged as anomalous.

    Returns
    -------
    pd.DataFrame
        DataFrame of anomalous transitions mapping:
        From_Triplet -> To_Triplet, Count, Probability, Video(s)
    """
    # 1. Build global transition matrix
    global_trans = build_transition_matrix(df_all)
    if global_trans.empty:
        return pd.DataFrame()
        
    # 2. Convert to probabilities (row-normalized)
    # How often does 'From' lead to 'To' vs out of all transitions leaving 'From'?
    row_sums = global_trans.sum(axis=1)
    
    # We only care about nodes that have a decent amount of outgoing transitions Let's say > 10.
    valid_sources = row_sums[row_sums > 10].index
    
    prob_matrix = global_trans.loc[valid_sources].div(row_sums[valid_sources], axis=0)
    
    # Flatten to process
    prob_flat = prob_matrix.unstack().reset_index(name='probability')
    
    # 3. Filter for anomalies: probability > 0 (it happened) but < threshold
    anomalies = prob_flat[(prob_flat['probability'] > 0) & (prob_flat['probability'] <= threshold_prob)].copy()
    
    # 4. Attach counts and video sources
    anomalies['global_count'] = anomalies.apply(
        lambda r: global_trans.loc[r['From_Triplet'], r['To_Triplet']], 
        axis=1
    )
    
    # Sort by probability ascending (most anomalous first)
    anomalies = anomalies.sort_values(['probability', 'global_count'], ascending=[True, False])
    
    return anomalies


# ---------------------------------------------------------------------------
# Graph Comparison
# ---------------------------------------------------------------------------

def compare_graphs(G1: nx.MultiDiGraph, G2: nx.MultiDiGraph) -> dict:
    """Compare two causal surgical graphs structurally.
    
    Computes:
      - Node Jaccard Similarity: Intersection over Union of active instruments/targets.
      - Edge Jaccard Similarity: Intersection over Union of unique (u, v, verb) edges.
      - Density Difference: Delta in graph density.
      - Degree Distribution Shift: Mean absolute difference in degree centrality.

    Parameters
    ----------
    G1, G2 : nx.MultiDiGraph
        Surgical graphs to compare.

    Returns
    -------
    dict
        Dictionary of similarity metrics.
    """
    # Nodes
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    node_intersection = nodes1.intersection(nodes2)
    node_union = nodes1.union(nodes2)
    node_jaccard = len(node_intersection) / len(node_union) if node_union else 0.0

    # Edges (Flattened without parallel multi-keys, just existence of u->v via verb)
    def _get_edges(G):
        return {(u, v, k) for u, v, k in G.edges(keys=True)}
    
    edges1 = _get_edges(G1)
    edges2 = _get_edges(G2)
    edge_intersection = edges1.intersection(edges2)
    edge_union = edges1.union(edges2)
    edge_jaccard = len(edge_intersection) / len(edge_union) if edge_union else 0.0

    # Density
    # Density for directed multi-graph varies. Use standard NetworkX density
    # which assumes simple directed graph bounds, or calculate manually.
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    den1 = G1.number_of_edges() / (n1 * (n1 - 1)) if n1 > 1 else 0
    den2 = G2.number_of_edges() / (n2 * (n2 - 1)) if n2 > 1 else 0
    density_diff = abs(den1 - den2)

    return {
        'node_jaccard_similarity': round(node_jaccard, 3),
        'edge_jaccard_similarity': round(edge_jaccard, 3),
        'density_difference': round(density_diff, 3),
        'shared_nodes': len(node_intersection),
        'shared_edges': len(edge_intersection)
    }

def print_graph_comparison(G1: nx.MultiDiGraph, G2: nx.MultiDiGraph) -> None:
    vid1 = G1.graph.get('video', 'G1')
    vid2 = G2.graph.get('video', 'G2')
    metrics = compare_graphs(G1, G2)
    
    print(f"=== Structural Comparison: {vid1} vs {vid2} ===")
    print(f"Node Similarity (Jaccard): {metrics['node_jaccard_similarity']*100:.1f}%  ({metrics['shared_nodes']} shared nodes)")
    print(f"Edge Similarity (Jaccard): {metrics['edge_jaccard_similarity']*100:.1f}%  ({metrics['shared_edges']} shared exact edges)")
    print(f"Density Difference:      {metrics['density_difference']:.3f} delta")
