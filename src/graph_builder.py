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

    # Helper to calculate average burst duration
    def compute_avg_burst(frames: list[int]) -> float:
        if not frames: return 0.0
        frames = sorted(frames)
        bursts = []
        current_burst = 1
        for i in range(1, len(frames)):
            # CholecT50 annotations might not be strictly +1 per frame if sample rate differs,
            # but assuming 1 fps parsed frames, +1 is consecutive.
            if frames[i] == frames[i-1] + 1:
                current_burst += 1
            else:
                bursts.append(current_burst)
                current_burst = 1
        bursts.append(current_burst)
        return sum(bursts) / len(bursts)

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
    edge_agg['avg_burst_duration'] = edge_agg['frame_list'].apply(compute_avg_burst)

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
            'avg_burst_duration': float(row['avg_burst_duration']),
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
# Visualization
# ---------------------------------------------------------------------------

def visualize_graph(
    G: nx.MultiDiGraph,
    output_path: str | Path | None = None,
    figsize: tuple = (16, 10),
    title: str | None = None,
) -> None:
    """Visualize the surgical action graph using matplotlib.

    Instruments and targets are color-coded and placed in a bipartite
    layout. Edge width is proportional to frequency, and verb labels
    are displayed on the edges.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph built by build_graph().
    output_path : str or Path, optional
        If provided, saves the plot to this path.
    figsize : tuple
        Figure size (width, height).
    title : str, optional
        Plot title. If None, uses the video name from graph metadata.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    instruments = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'instrument']
    targets = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']

    # Bipartite layout: instruments on the left, targets on the right
    pos = {}
    for i, node in enumerate(sorted(instruments)):
        pos[node] = (-1, -i * 1.2)
    for i, node in enumerate(sorted(targets)):
        pos[node] = (1, -i * 0.8)

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in instruments:
            node_colors.append('#2196F3')  # blue
            node_sizes.append(2000)
        else:
            node_colors.append('#4CAF50')  # green
            node_sizes.append(1800)

    # Edge widths proportional to log-frequency
    import math
    edge_data = []
    for u, v, k, d in G.edges(data=True, keys=True):
        freq = d.get('frequency', 1)
        edge_data.append((u, v, k, freq))

    max_freq = max(e[3] for e in edge_data) if edge_data else 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='white',
        linewidths=2,
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=10,
        font_weight='bold',
        font_color='white',
    )

    # Draw edges with varying width
    for u, v, verb, freq in edge_data:
        width = 1 + 4 * (freq / max_freq)
        alpha = 0.4 + 0.5 * (freq / max_freq)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color='#455A64',
            arrows=True,
            arrowsize=15,
            connectionstyle='arc3,rad=0.1',
        )
        # Edge label (verb + frequency)
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        offset = 0.08 * (hash(verb) % 5 - 2)
        ax.text(
            mid_x, mid_y + offset,
            f'{verb} ({freq})',
            fontsize=7,
            ha='center',
            va='center',
            color='#37474F',
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
        )

    # Legend
    inst_patch = mpatches.Patch(color='#2196F3', label=f'Instruments ({len(instruments)})')
    tgt_patch = mpatches.Patch(color='#4CAF50', label=f'Targets ({len(targets)})')
    ax.legend(handles=[inst_patch, tgt_patch], loc='upper right', fontsize=11)

    video_name = G.graph.get('video', 'Unknown')
    if title is None:
        title = f'Surgical Action Graph — {video_name}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'  Saved {output_path.name}')

    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive Visualization (pyvis)
# ---------------------------------------------------------------------------

def visualize_graph_interactive(
    G: nx.MultiDiGraph,
    output_path: str | Path | None = None,
    title: str | None = None,
    height: str = '800px',
    width: str = '100%',
) -> Path | None:
    """Generate an interactive HTML graph viewable in any web browser.

    Instruments and targets are color-coded with distinct shapes and
    descriptive labels (e.g. "Instrument (grasper)", "Target (gallbladder)").

    Parameters
    ----------
    G : nx.MultiDiGraph
        Graph built by build_graph().
    output_path : str or Path, optional
        Where to save the HTML file. If None, saves to
        outputs/graphs/interactive/<video>_interactive.html.
    title : str, optional
        Graph title. If None, auto-generated from graph metadata.
    height : str
        CSS height for the canvas (default '800px').
    width : str
        CSS width for the canvas (default '100%').

    Returns
    -------
    Path or None
        Path to saved HTML file.
    """
    from pyvis.network import Network

    video_name = G.graph.get('video', 'Unknown')
    if title is None:
        title = f'Surgical Action Graph — {video_name}'

    # Separate node types
    instruments = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'instrument']
    targets = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'target']

    # Create pyvis network
    net = Network(
        height=height,
        width=width,
        directed=True,
        notebook=False,
        bgcolor='#1a1a2e',
        font_color='white',
    )

    # Color scheme
    INSTRUMENT_COLOR = '#2196F3'  # blue
    TARGET_COLOR = '#4CAF50'      # green

    # --- Add Instrument nodes ---
    for node in sorted(instruments):
        node_id = G.nodes[node].get('id', '?')
        label = f'Instrument\n({node})'
        tooltip = (
            f'<b>Instrument: {node}</b><br>'
            f'ID: {node_id}<br>'
            f'Type: Instrument<br>'
            f'Connections: {G.degree(node)}'
        )
        net.add_node(
            node,
            label=label,
            title=tooltip,
            color=INSTRUMENT_COLOR,
            shape='dot',
            size=35,
            font={'size': 14, 'color': 'white', 'face': 'arial', 'strokeWidth': 2, 'strokeColor': '#000'},
            borderWidth=3,
            borderWidthSelected=5,
        )

    # --- Add Target nodes ---
    for node in sorted(targets):
        node_id = G.nodes[node].get('id', '?')
        label = f'Target\n({node})'
        tooltip = (
            f'<b>Target: {node}</b><br>'
            f'ID: {node_id}<br>'
            f'Type: Target<br>'
            f'Connections: {G.degree(node)}'
        )
        net.add_node(
            node,
            label=label,
            title=tooltip,
            color=TARGET_COLOR,
            shape='diamond',
            size=30,
            font={'size': 14, 'color': 'white', 'face': 'arial', 'strokeWidth': 2, 'strokeColor': '#000'},
            borderWidth=3,
            borderWidthSelected=5,
        )

    # --- Add Edges ---
    max_freq = 1
    for u, v, k, d in G.edges(data=True, keys=True):
        freq = d.get('frequency', 1)
        if freq > max_freq:
            max_freq = freq

    for u, v, k, d in G.edges(data=True, keys=True):
        verb = d.get('verb', k)
        freq = d.get('frequency', 1)
        first_f = d.get('first_frame', '?')
        last_f = d.get('last_frame', '?')
        triplet_label = d.get('triplet_label', '')

        edge_width = 1 + 5 * (freq / max_freq)
        edge_label = f'{verb} ({freq})'
        tooltip = (
            f'<b>{verb}</b><br>'
            f'Frequency: {freq} frames<br>'
            f'Triplet: {triplet_label}<br>'
            f'Frames: {first_f} → {last_f}<br>'
            f'Avg burst: {d.get("avg_burst_duration", 0):.1f}'
        )

        net.add_edge(
            u, v,
            label=edge_label,
            title=tooltip,
            width=edge_width,
            color={'color': '#78909C', 'highlight': '#FF9800', 'hover': '#FF9800'},
            arrows='to',
            font={'size': 10, 'color': '#B0BEC5', 'strokeWidth': 0, 'align': 'top'},
            smooth={'type': 'curvedCW', 'roundness': 0.2},
        )

    # --- Physics & interaction options ---
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.008,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "iterations": 150
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 100,
        "zoomView": true
      },
      "edges": {
        "smooth": {
          "type": "curvedCW",
          "roundness": 0.2
        }
      }
    }
    """)

    # --- Determine output path ---
    if output_path is None:
        output_path = Path('outputs/graphs/interactive') / f'{video_name}_interactive.html'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Add legend + title via custom HTML injection ---
    extra_html = f"""
    <div style="
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 14px 24px; text-align: center;
        font-family: Arial, sans-serif; color: white;
        border-bottom: 2px solid #2196F3;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    ">
        <span style="font-size: 20px; font-weight: bold; letter-spacing: 0.5px;">
            🔬 {title}
        </span>
    </div>
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: rgba(26, 26, 46, 0.92); border: 1px solid #333;
        border-radius: 12px; padding: 16px 20px;
        font-family: Arial, sans-serif; color: white; font-size: 13px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); backdrop-filter: blur(8px);
    ">
        <div style="font-weight: bold; font-size: 15px; margin-bottom: 10px;
                    border-bottom: 1px solid #444; padding-bottom: 6px;">
            🔬 Legend
        </div>
        <div style="margin-bottom: 6px;">
            <span style="display: inline-block; width: 14px; height: 14px;
                        background: {INSTRUMENT_COLOR}; border-radius: 50%;
                        vertical-align: middle; margin-right: 8px;"></span>
            <b>Instrument</b> — {len(instruments)} nodes
        </div>
        <div style="margin-bottom: 10px;">
            <span style="display: inline-block; width: 14px; height: 14px;
                        background: {TARGET_COLOR}; transform: rotate(45deg);
                        vertical-align: middle; margin-right: 8px;"></span>
            <b>Target</b> — {len(targets)} nodes
        </div>
        <div style="font-size: 11px; color: #aaa;">
            Edges: {G.number_of_edges()} | Drag to rearrange | Scroll to zoom
        </div>
    </div>
    """

    # Save and inject legend
    net.save_graph(str(output_path))

    # Inject legend HTML into the saved file
    with open(output_path, 'r') as f:
        html_content = f.read()
    html_content = html_content.replace('</body>', extra_html + '\n</body>')
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f'  Saved interactive graph: {output_path}')
    return output_path


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

