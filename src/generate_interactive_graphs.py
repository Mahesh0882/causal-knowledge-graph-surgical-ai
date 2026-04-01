"""
Generate Interactive Graphs for All Parsed Videos
===================================================
Loads parsed triplet CSVs, builds NetworkX graphs, and exports
interactive HTML files viewable in any web browser.

Usage:
    python src/generate_interactive_graphs.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.graph_builder import build_graph, visualize_graph_interactive, print_graph_stats


def main():
    csv_dir = project_root / 'outputs' / 'parsed_triplets'
    output_dir = project_root / 'outputs' / 'graphs' / 'interactive'

    # Find all per-video CSVs (skip all_triplets.csv)
    csv_files = sorted([
        f for f in csv_dir.glob('*_triplets.csv')
        if f.stem != 'all_triplets'
    ])

    if not csv_files:
        print(f'ERROR: No triplet CSVs found in {csv_dir}')
        sys.exit(1)

    print(f'Found {len(csv_files)} video CSV file(s) in {csv_dir}\n')

    generated = []
    for csv_path in csv_files:
        video_name = csv_path.stem.replace('_triplets', '')
        print(f'--- {video_name} ---')

        df = pd.read_csv(csv_path)
        G = build_graph(df, video_name=video_name)
        print_graph_stats(G)

        out_path = output_dir / f'{video_name}_interactive.html'
        visualize_graph_interactive(G, output_path=out_path)
        generated.append(out_path)
        print()

    # Summary
    print('=' * 60)
    print(f'Generated {len(generated)} interactive graph(s):')
    for p in generated:
        print(f'  → {p}')
    print(f'\nOpen any HTML file in your browser to explore the graph!')
    print('=' * 60)


if __name__ == '__main__':
    main()
