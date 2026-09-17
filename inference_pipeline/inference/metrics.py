import numpy as np

def compute_reconstruction_metrics(W_true, W_inferred):
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)  # ignore diagonal
    
    gt_edges = (W_true[mask] != 0).astype(int)
    inf_edges = (W_inferred[mask] != 0).astype(int)
    
    tp = np.sum((gt_edges == 1) & (inf_edges == 1))
    fp = np.sum((gt_edges == 0) & (inf_edges == 1))
    fn = np.sum((gt_edges == 1) & (inf_edges == 0))
    tn = np.sum((gt_edges == 0) & (inf_edges == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    shd = fp + fn
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "TPR (Recall)": tpr, "FPR": fpr,
        "Precision": precision, "F1-Score": f1,
        "SHD": shd
    }

import matplotlib.pyplot as plt
import seaborn as sns

def plot_probabilistic_network(W_true, edge_probabilities, save_path="images/probabilistic_network.png"):
    """
    Plots a side-by-side comparison of the True binary adjacency matrix
    and the Inferred marginal edge probabilities.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    true_adj = (W_true != 0).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(true_adj, ax=axes[0], cmap="Blues", cbar=False, square=True)
    axes[0].set_title("True Network (Binary)")
    axes[0].set_xlabel("Source Gene")
    axes[0].set_ylabel("Target Gene")
    
    sns.heatmap(edge_probabilities, ax=axes[1], cmap="Reds", vmin=0, vmax=edge_probabilities.max(), square=True)
    axes[1].set_title("Inferred Edge Probabilities")
    axes[1].set_xlabel("Source Gene")
    axes[1].set_ylabel("Target Gene")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved probabilistic network plot to {save_path}")
    plt.close()

def plot_inferred_vs_true_am(W_true, W_inferred, save_path="images/am_comparison.png"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    vmax = max(np.abs(W_true).max(), np.abs(W_inferred).max())
    
    sns.heatmap(W_true, ax=axes[0], cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, square=True)
    axes[0].set_title("True Network Weights")
    
    sns.heatmap(W_inferred, ax=axes[1], cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax, square=True)
    axes[1].set_title("Inferred Network Weights (Last Sample)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved AM comparison plot to {save_path}")
    plt.close()
