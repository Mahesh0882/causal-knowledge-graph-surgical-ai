# AAAI Research Roadmap — Weekly Execution Plan

**Goal:** Transform the current descriptive graph analysis into a predictive, explainable surgical workflow system targeting AAAI submission.

**Core Research Question:** *Can causal graph representations improve surgical workflow prediction and interpretability compared to standard sequence-only or video-based models?*

---

## Current State (Pilot / What We Have)

| Asset | Status |
|---|---|
| Triplet parsing (`triplet_parser.py`) | ✅ Complete — parses CholecT50 JSON → tidy DataFrames |
| Graph construction (`graph_builder.py`) | ✅ Complete — builds NetworkX MultiDiGraph per video |
| Transition matrices (`temporal_analysis.py`) | ✅ Complete — co-occurrence + frame-to-frame transitions |
| Rule-based anomaly detection (`advanced_analytics.py`) | ✅ Complete — threshold-based (< 5% prob) |
| Phase subgraphs / edge cases (`edge_cases.py`) | ✅ Complete |
| Interactive visualization (`generate_interactive_graphs.py`) | ✅ Complete — pyvis HTML |
| **CholecT50 data (50 video JSONs)** | ✅ Available in `/CholecT50/labels/` |
| Learnable predictive models | ❌ Not started |
| Full-dataset processing pipeline | ❌ Only 3 videos processed |
| Baselines (LSTM, Transformer) | ❌ Not started |
| Quantitative evaluation | ❌ Not started |

---

## Weekly Plan (8 Weeks)

### Week 1 — Full Dataset Pipeline & Feature Engineering
**Theme:** Scale from 3 videos → all 45+ videos, build the training data foundation.

- [x] **Parse all 50 CholecT50 videos** through `triplet_parser.py` and save CSVs
- [x] **Build graphs for all videos** using `graph_builder.py`
- [x] **Create train/val/test splits** (suggest: 30/7/8 or follow CholecT45 official splits if available)
- [x] **Build temporal graph sequence dataset:**
  - For each video, create a sequence of graph snapshots (one per frame or per N-frame window)
  - Each snapshot = set of active (instrument, verb, target) triplets
  - Store as structured format (e.g., list of `(frame_id, graph_state)` tuples)
- [x] **Feature extraction pipeline:**
  - Node embeddings: one-hot or learned ID-based for instruments (6 classes), verbs (10 classes), targets (15 classes)
  - Edge features: verb type (categorical), frequency, burst duration
  - Global graph features: density, node count, phase ID
- [x] **Verify:** All 50 videos parse without error; spot-check 5 random videos visually

> [!IMPORTANT]
> This week establishes the entire data pipeline. Everything downstream depends on clean, complete data.

---

### Week 2 — Task 1: Next Action Prediction (Baselines)
**Theme:** Build sequence-only baselines so we have something to beat.

- [x] **Define the prediction task formally:**
  - Input: triplet sequence up to frame *t* (window of last *k* frames, e.g. k=10)
  - Output: next triplet(s) at frame *t+1*
  - Multi-label: multiple triplets can be active simultaneously
- [x] **Implement Baseline 1 — LSTM:**
  - Encode each frame as a multi-hot vector over all 100 triplet classes
  - LSTM processes the window → predicts next multi-hot vector
  - Loss: Binary Cross-Entropy (multi-label)
- [x] **Implement Baseline 2 — Transformer (sequence-only):**
  - Same input encoding
  - Positional encoding + self-attention over the frame sequence
  - Predict next triplet distribution
- [x] **Implement Baseline 3 — Markov / Transition Matrix:**
  - Use the transition matrix from `temporal_analysis.py` as a statistical baseline
  - P(next | current) directly from the global transition matrix
- [x] **Training loop + evaluation script:**
  - Metrics: Top-1 Accuracy, Top-5 Accuracy, F1-Score (macro), Mean Average Precision
- [x] **Verify:** Train baselines on the train split, report val metrics

> [!TIP]
> Use PyTorch for all learnable models. Keep the training loop modular — we'll reuse it for graph models.

---

### Week 3 — Task 1: Next Action Prediction (Graph-Based Model)
**Theme:** Build the core Temporal GNN model — this is the main contribution.

- [ ] **Design the Temporal Graph Neural Network (T-GNN):**
  - **Per-frame GNN:** Each frame's active triplets form a bipartite graph (instruments → targets via verb edges)
  - **GNN Layer:** GCN or GAT to aggregate neighbor information (instrument ↔ target interaction)
  - **Temporal Module:** Feed the sequence of graph embeddings into a GRU/LSTM or Temporal Transformer
  - **Prediction Head:** MLP over the final temporal embedding → predict next frame's triplets
- [ ] **Implement using PyTorch Geometric (PyG):**
  - `Data` / `Batch` objects for each frame's graph
  - Custom `TemporalGraphDataset` that yields sequences of graphs
- [ ] **Train and compare against Week 2 baselines**
- [ ] **Ablation studies:**
  - T-GNN vs GNN-only (no temporal) vs Temporal-only (no graph structure)
  - Effect of window size *k*
- [ ] **Verify:** Graph model outperforms at least one baseline on val set

> [!IMPORTANT]
> **This is Contribution C2** — "A graph-based predictive model that outperforms sequence-only and video-only baselines in next-action prediction."

---

### Week 4 — Task 2: Anomaly Detection (Learnable)
**Theme:** Upgrade the rule-based anomaly detection to a learning-based approach.

- [ ] **Design anomaly detection framework:**
  - **Approach A — Likelihood-based:** Use the T-GNN's predicted distribution. If the actual next action has very low predicted probability → flag as anomaly
  - **Approach B — Reconstruction-based:** Train an autoencoder on graph sequences; anomalies = high reconstruction error
  - **Approach C — Energy-based:** Score graph transitions using an energy function
- [ ] **Create anomaly labels:**
  - Since CholecT50 has no explicit anomaly labels, use proxy strategy:
    - *Statistical anomalies*: transitions below global 5th percentile (reuse `advanced_analytics.py` logic)
    - *Held-out video anomalies*: train on majority of videos, treat rare per-video patterns as anomalies
  - Or: manually annotate a small subset with domain expert (discuss with advisor)
- [ ] **Implement the chosen approach** (recommend Approach A first — most natural extension)
- [ ] **Evaluation:**
  - AUROC, Precision, Recall, F1 for anomaly classification
  - Compare against the current rule-based threshold method
- [ ] **Verify:** Learning-based anomaly detection improves over rule-based baseline

> [!WARNING]
> Anomaly labels are the hardest part here. If you can get even a small expert-annotated set (~5 videos), it dramatically strengthens the paper. Discuss with your advisor.

---

### Week 5 — Task 3: Workflow Reasoning & Explainability
**Theme:** Make the model interpretable — explain *why* predictions are made via graph paths.

- [ ] **Phase prediction task:**
  - Given current graph state → predict current surgical phase
  - Use phase labels from CholecT50 annotations (already parsed in `triplet_parser.py`)
- [ ] **Graph path explanations:**
  - For a given prediction, extract the most influential graph paths (instrument → verb → target chains)
  - Visualize attention weights from GAT or Transformer layers
  - Generate human-readable explanations: "Predicted *clip → clip → cystic_duct* because *grasper → retract → gallbladder* was active (similar to 85% of training cases)"
- [ ] **Deviation detection:**
  - Compare current video's graph evolution against the "average" phase-specific graph from training data
  - Flag when the current state diverges significantly (use `compare_graphs()` from `advanced_analytics.py`)
- [ ] **Case studies:**
  - Pick 3–5 interesting videos and create detailed visual walkthroughs
  - Show: actual vs predicted workflow, detected anomalies, graph explanations
- [ ] **Verify:** Phase prediction accuracy reported; at least 3 case studies with compelling visualizations

---

### Week 6 — Experiments & Full Evaluation
**Theme:** Run all final experiments, create comparison tables, generate all figures.

- [ ] **Final training runs** with best hyperparameters on all 3 tasks
- [ ] **Cross-validation or multiple random seeds** (at least 3 runs for error bars)
- [ ] **Compile results tables:**

| Model | Top-1 Acc | Top-5 Acc | F1 (macro) | mAP |
|---|---|---|---|---|
| Markov Baseline | — | — | — | — |
| LSTM | — | — | — | — |
| Transformer | — | — | — | — |
| **T-GNN (Ours)** | — | — | — | — |

- [ ] **Anomaly detection results table** (AUROC, Precision, Recall)
- [ ] **Generate publication-quality figures:**
  - Architecture diagram (T-GNN pipeline)
  - Graph visualization examples (reuse interactive graphs, make static versions)
  - Attention heatmaps / explanation visualizations
  - t-SNE/UMAP of learned graph embeddings
- [ ] **Verify:** All numbers reproducible from saved checkpoints; figures export-ready

---

### Week 7 — Paper Writing
**Theme:** Draft the AAAI paper (8 pages, AAAI format).

- [ ] **Paper structure:**
  1. **Abstract** (250 words)
  2. **Introduction** — motivation, research question, contributions list
  3. **Related Work** — surgical workflow modeling, GNNs for surgery, anomaly detection
  4. **Method** — graph formulation, T-GNN architecture, training procedure
  5. **Experiments** — dataset, baselines, metrics, results tables
  6. **Analysis** — ablations, case studies, graph explanations
  7. **Conclusion + Future Work** — multimodal extension mention
- [ ] **Write explicit contribution claims (pick 2–4 from the guide):**
  - C1: Temporal causal graph formulation from structured triplets
  - C2: Graph-based predictive model outperforming baselines
  - C3: Learning-based anomaly detection for surgical deviations
  - C4 (optional): Interpretable reasoning via graph paths
- [ ] **Literature review** — identify 15–20 key papers to cite
- [ ] **Verify:** Draft reviewed by advisor

---

### Week 8 — Polish, Reproducibility & Submission Prep
**Theme:** Final polish, code cleanup, supplementary materials.

- [ ] **Code cleanup:**
  - Clean `requirements.txt` to include PyTorch, PyG, and new dependencies
  - Add `train.py`, `evaluate.py` scripts with CLI arguments
  - Update `README.md` with full reproduction instructions
- [ ] **Reproducibility package:**
  - All model checkpoints saved
  - Config files for hyperparameters (YAML)
  - Single script to reproduce all tables/figures
- [ ] **Supplementary materials:**
  - Extended results tables
  - Additional case studies
  - Dataset statistics appendix
- [ ] **AAAI checklist review** (from the guide):
  - [x] Full dataset used (45+ videos)
  - [x] At least one predictive task evaluated
  - [x] Learnable model included
  - [x] Strong baselines compared
  - [x] Quantitative evaluation with standard metrics
  - [x] 2–4 explicit contributions stated
- [ ] **Push final code to `vsi-lab/causal-knowledge-graph-surgical-ai`**
- [ ] **Verify:** Advisor sign-off, code runs end-to-end from scratch

---

## Reusable Assets from Pilot (from the AAAI Guide)

These existing components will be **absorbed into the learning pipeline** as features/priors:

| Pilot Asset | How It's Reused |
|---|---|
| Triplet representation | Input encoding for all models |
| Transition matrices | Markov baseline + prior for T-GNN initialization |
| Co-occurrence patterns | Feature input for multi-label prediction |
| Phase-based subgraphs | Phase prediction task ground truth |
| Initial anomaly concept | Proxy labels for anomaly detection |

---

## Key Dependencies to Add

```
torch>=2.0
torch-geometric>=2.4
torch-scatter
torch-sparse
scikit-learn
tensorboard
pyyaml
tqdm
```

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| T-GNN doesn't beat baselines | Add graph attention (GAT), try different temporal modules, adjust window size |
| No real anomaly labels | Use statistical proxy + request small expert annotation set |
| Compute limitations (Mac) | Use Google Colab Pro / university GPU cluster for training |
| CholecT45 vs CholecT50 confusion | CholecT45 is the standard subset; verify which videos are in the canonical split |
| Timeline too tight | Weeks 5-6 can overlap; paper writing can start incrementally |
