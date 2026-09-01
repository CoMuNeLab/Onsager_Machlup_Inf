# Visualizing Graph-Tool GRN Inference Results (`gene_expr`)

Inspired by the benchmark figures in `PROJECTS/BioInf/new_proj/figures/test_pipeline_plots`, we have created an automated visualization suite for our `graph-tool` Gene Expression Bayesian Inference extension (`gene_expr`).

The generating script is located at:
- [`plot_graph_tool_inference_results.py`](file:///home/francesco/Universita/PhD/PROJECTS/InferenceBio/plot_graph_tool_inference_results.py)

All output plots are saved at 300 DPI in:
- [`figures/`](file:///home/francesco/Universita/PhD/PROJECTS/InferenceBio/figures/)

---

## 1. Figure 1: Ground-Truth Adjacency vs. Posterior Edge Probability Heatmaps
**File**: `figures/fig01_ground_truth_vs_inferred_heatmaps.png`

### Description
A 1×2 side-by-side comparative heatmap panel inspired by `article_rep_01_heatmap_boxes.png`:
- **Left Panel**: The ground-truth 10-gene signed regulatory adjacency matrix ($W_{\text{true}}$) with activating edges ($+1$) shown in red and repressing edges ($-1$) in blue.
- **Right Panel**: The inferred posterior edge probability matrix ($P_{\text{edge}}$) across MCMC sweeps, where each cell represents the empirical posterior frequency $P(W_{ij} \neq 0)$.
- **Rectangular Bounding Boxes**: Ground-truth edges are overlaid on the posterior heatmap using **thick red rectangular bounding boxes** for true activations and **blue rectangular bounding boxes** for true repressions, enabling instant visual assessment of structure recovery.

---

## 2. Figure 2: MCMC Posterior Description Length (Entropy) Convergence
**File**: `figures/fig02_mcmc_entropy_convergence.png`

### Description
Inspired by `article_rep_04_mcmc_energy.png` and `fig07_adaptive_convergence.png`:
- Plots the trajectory of the total **Posterior Description Length ($\Sigma$) / Entropy** across MCMC sweeps.
- Highlights the burn-in phase boundary (vertical dashed line at Sweep 30) where the sampler completes transient exploration and transitions to sampling the stationary posterior distribution of network topologies and continuous parameters ($\theta$).

---

## 3. Figure 3: Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves
**File**: `figures/fig03_roc_pr_curves.png`

### Description
Inspired by `article_rep_03_roc_comparison.png` and `fig02_priors_roc_curves.png`:
- **Left Panel**: Receiver Operating Characteristic (ROC) curve plotting True Positive Rate (TPR / Recall) against False Positive Rate (FPR) across all possible classification thresholds on the edge posterior probabilities $P_{\text{edge}}$, including the **Area Under the ROC Curve (AUROC)** score.
- **Right Panel**: Precision-Recall Curve (PRC) plotting Precision against Recall, including the **Area Under the PR Curve (AUPRC)** and the baseline random-guess precision line.

---

## 4. Figure 4: Posterior Edge Probability Histogram / Density
**File**: `figures/fig04_link_probability_histogram.png`

### Description
Inspired by `article_rep_02_link_probs.png`:
- A Kernel Density Estimation (KDE) and density distribution plot comparing the inferred posterior edge probabilities $P(W_{ij} \neq 0)$ for **Ground-Truth Regulatory Interactions** ($W_{\text{true}} \neq 0$, red density) versus **True Non-Edges / Null Interactions** ($W_{\text{true}} = 0$, blue density).
- Visualizes the discriminative power of the Onsager–Machlup continuous path likelihood in separating signal from stochastic noise.

---

## Usage

To regenerate all figures from the command line:

```bash
/home/francesco/anaconda3/envs/gt/bin/python plot_graph_tool_inference_results.py
```
