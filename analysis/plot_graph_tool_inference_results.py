#!/usr/bin/env python3
"""
Visualization Script for Graph-Tool GRN Inference Results (`gene_expr`),
taking inspiration from the benchmark publication plots in PROJECTS/BioInf/new_proj.

Generates 4 publication-ready figures in data/figures (or ./figures):
  1. fig01_ground_truth_vs_inferred_heatmaps.png:
     - Side-by-side comparison of W_true and Inferred Posterior Edge Probability P_edge
       with red/blue bounding boxes highlighting ground-truth edges.
  2. fig02_mcmc_entropy_convergence.png:
     - MCMC Posterior Description Length (Entropy) vs. Sweep iteration.
  3. fig03_roc_pr_curves.png:
     - ROC curve (AUROC) and Precision-Recall curve (AUPRC) on edge reconstruction.
  4. fig04_link_probability_histogram.png:
     - Posterior edge probability distribution for True Edges vs. Non-Edges.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add project root to sys.path
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from analysis.run_onsager_graph_tool_pipeline import (
    get_benchmark_10gene_network,
    simulate_grn_sde_trajectory,
)

# Publication styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})

def create_output_dir(dir_path="images"):
    out_dir = os.path.join(_proj_dir, dir_path)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def run_inference_and_sample_posterior(W_true, n_sweeps=80, burn_in=30):
    """
    Simulates time-series data and performs MCMC sweeps, returning:
      - entropy_history: list of entropy values across sweeps
      - P_edge: posterior edge probability matrix (N x N)
    """
    N = W_true.shape[0]
    print(f"[1/3] Simulating SDE trajectories for N={N} genes...")
    s, dt_meas = simulate_grn_sde_trajectory(
        W_true, T_total=200.0, dt=0.01, dt_meas=0.5,
        d=0.5, V=2.0, noise_std=0.15, seed=424
    )
    
    print("[2/3] Instantiating GeneExprBlockState and running MCMC sweeps...")
    state = GeneExprBlockState(s, dt=dt_meas)
    
    entropy_history = [state.entropy()]
    edge_counts = np.zeros((N, N), dtype=float)
    n_samples = 0
    
    for sweep in range(1, n_sweeps + 1):
        dS, nmoves = state.mcmc_sweep(niter=15)
        S_cur = state.entropy()
        entropy_history.append(S_cur)
        
        if sweep > burn_in:
            g = state.get_graph()
            for e in g.edges():
                src, tgt = int(e.source()), int(e.target())
                if src != tgt: # record directed edge
                    edge_counts[tgt, src] += 1.0
            n_samples += 1
            
        if sweep % 10 == 0 or sweep == 1:
            print(f"      Sweep {sweep:2d}/{n_sweeps}: Entropy = {S_cur:8.4f} (dS = {dS:8.4f}, moves = {nmoves:4d})")
            
    P_edge = edge_counts / max(1, n_samples)
    return entropy_history, P_edge

def plot_heatmaps_with_boxes(W_true, P_edge, out_dir):
    """
    Figure 1: Side-by-side comparison of W_true and P_edge with red/blue bounding boxes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    N = W_true.shape[0]
    
    # Left: W_true
    ax0 = axes[0]
    sns.heatmap(W_true, ax=ax0, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                cbar_kws={"label": "Regulatory Interaction $W_{ij}$"},
                linewidths=0.5, linecolor="gray", square=True)
    ax0.set_title("Ground-Truth Regulatory Network ($W_{\\text{true}}$)")
    ax0.set_xlabel("Source Gene ($j$)")
    ax0.set_ylabel("Target Gene ($i$)")
    
    # Right: Posterior Edge Probabilities with Bounding Boxes
    ax1 = axes[1]
    sns.heatmap(P_edge, ax=ax1, cmap="YlGnBu", vmin=0, vmax=1,
                cbar_kws={"label": "Posterior Probability $P(W_{ij} \\neq 0)$"},
                linewidths=0.5, linecolor="lightgray", square=True)
    ax1.set_title("Inferred Edge Posterior ($P_{\\text{edge}}$) + True Boxes")
    ax1.set_xlabel("Source Gene ($j$)")
    ax1.set_ylabel("Target Gene ($i$)")
    
    # Add colored rectangular bounding boxes for ground-truth edges
    for i in range(N):
        for j in range(N):
            val = W_true[i, j]
            if val != 0 and i != j:
                color = "red" if val > 0 else "blue"
                rect = patches.Rectangle((j, i), 1, 1, fill=False,
                                         edgecolor=color, linewidth=2.5)
                ax1.add_patch(rect)
                
    # Add legend for boxes
    red_patch = patches.Patch(edgecolor="red", facecolor="none", linewidth=2.5, label="True Activation ($+1$)")
    blue_patch = patches.Patch(edgecolor="blue", facecolor="none", linewidth=2.5, label="True Repression ($-1$)")
    ax1.legend(handles=[red_patch, blue_patch], loc="upper right", bbox_to_anchor=(1.35, 1.15))
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig01_ground_truth_vs_inferred_heatmaps.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 1: {save_path}")

def plot_mcmc_entropy(entropy_history, burn_in, out_dir):
    """
    Figure 2: MCMC Posterior Description Length (Entropy) Convergence.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sweeps = np.arange(len(entropy_history))
    
    ax.plot(sweeps, entropy_history, "o-", color="#2b5c8f", linewidth=2, markersize=5, label="MCMC Sweep Entropy $\\Sigma$")
    ax.axvline(x=burn_in, color="#d95f02", linestyle="--", linewidth=2, label="Burn-In Boundary")
    
    ax.set_title("Bayesian Network Inference: MCMC Description Length Convergence")
    ax.set_xlabel("MCMC Sweep Iteration")
    ax.set_ylabel("Posterior Description Length $\\Sigma$ (Entropy)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig02_mcmc_entropy_convergence.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 2: {save_path}")

def plot_roc_pr_curves(W_true, P_edge, out_dir):
    """
    Figure 3: ROC curve (AUROC) and Precision-Recall curve (AUPRC).
    """
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    
    y_true = (W_true[mask] != 0).astype(int)
    y_score = P_edge[mask]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    baseline_pr = np.mean(y_true)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: ROC
    ax0 = axes[0]
    ax0.plot(fpr, tpr, color="#d95f02", linewidth=2.5, label=f"Graph-Tool Posterior (AUC = {roc_auc:.3f})")
    ax0.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Guess (AUC = 0.500)")
    ax0.set_title("Receiver Operating Characteristic (ROC)")
    ax0.set_xlabel("False Positive Rate (FPR)")
    ax0.set_ylabel("True Positive Rate (TPR)")
    ax0.legend(loc="lower right")
    ax0.grid(True, linestyle=":", alpha=0.7)
    
    # Right: PR
    ax1 = axes[1]
    ax1.plot(recall, precision, color="#1b9e77", linewidth=2.5, label=f"Graph-Tool Posterior (AUPRC = {pr_auc:.3f})")
    ax1.axhline(baseline_pr, color="k", linestyle="--", linewidth=1.5, label=f"Baseline Precision ({baseline_pr:.3f})")
    ax1.set_title("Precision-Recall Curve (PRC)")
    ax1.set_xlabel("Recall (TPR)")
    ax1.set_ylabel("Precision")
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig03_roc_pr_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 3: {save_path} (AUROC={roc_auc:.3f}, AUPRC={pr_auc:.3f})")

def plot_link_probability_histogram(W_true, P_edge, out_dir):
    """
    Figure 4: Distribution of posterior edge probabilities for true edges vs non-edges.
    """
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    
    y_true = (W_true[mask] != 0).astype(int)
    y_score = P_edge[mask]
    
    pos_probs = y_score[y_true == 1]
    neg_probs = y_score[y_true == 0]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.kdeplot(pos_probs, fill=True, color="#d95f02", label=f"Ground-Truth Edges (n={len(pos_probs)})", ax=ax, alpha=0.5, bw_adjust=0.5)
    sns.kdeplot(neg_probs, fill=True, color="#2b5c8f", label=f"Non-Edges (n={len(neg_probs)})", ax=ax, alpha=0.5, bw_adjust=0.5)
    
    ax.set_title("Posterior Link Probability Density: True Edges vs. Non-Edges")
    ax.set_xlabel("Inferred Posterior Probability $P(W_{ij} \\neq 0)$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig04_link_probability_histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 4: {save_path}")

def main():
    print("=" * 72)
    print("GENERATING INFERENCE VISUALIZATIONS FOR GRAPH-TOOL GRN EXTENSION")
    print("=" * 72)
    
    out_dir = create_output_dir("images")
    print(f"Output directory: {out_dir}\n")
    
    W_true = get_benchmark_10gene_network()
    entropy_history, P_edge = run_inference_and_sample_posterior(
        W_true, n_sweeps=80, burn_in=30
    )
    
    print("\n[3/3] Generating publication figures...")
    plot_heatmaps_with_boxes(W_true, P_edge, out_dir)
    plot_mcmc_entropy(entropy_history, burn_in=30, out_dir=out_dir)
    plot_roc_pr_curves(W_true, P_edge, out_dir)
    plot_link_probability_histogram(W_true, P_edge, out_dir)
    
    print("\n=" * 72)
    print("SUCCESS: All 4 publication figures generated cleanly in ./figures!")
    print("=" * 72)

if __name__ == "__main__":
    main()
