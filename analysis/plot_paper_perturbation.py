import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure we can import the metrics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from perturbation_metrics import evaluate_shift_correlation, evaluate_calibration, evaluate_multimodal_kde
from scipy.stats import spearmanr

def load_julia_binary(filepath, shape):
    """Loads a Julia-exported binary file (column-major) into a numpy array."""
    data = np.fromfile(filepath, dtype=np.float64)
    return data.reshape(shape, order='F').T

def main():
    print("Loading data...")
    N_nodes = 10
    bytes_per_float = 8
    
    N_samples = int(os.path.getsize("results/stat_inferred_pert.bin") / (N_nodes * bytes_per_float))
    N_true_samples = int(os.path.getsize("results/stat_true_pert.bin") / (N_nodes * bytes_per_float))
    
    stat_inferred_pert = load_julia_binary("results/stat_inferred_pert.bin", (N_nodes, N_samples))
    stat_inferred_unpert = load_julia_binary("results/stat_inferred_unpert.bin", (N_nodes, N_samples))
    stat_true_pert_ens = load_julia_binary("results/stat_true_pert.bin", (N_nodes, N_true_samples))
    stat_true_unpert_ens = load_julia_binary("results/stat_true_unpert.bin", (N_nodes, N_true_samples))
    
    true_pert = stat_true_pert_ens.mean(axis=0)
    true_unpert = stat_true_unpert_ens.mean(axis=0)
    
    ko_idx = 0
    
    # We found via graph-tool that sample 50 has the absolute lowest entropy (MAP estimate)
    best_idx = 50
    
    delta_true = true_pert - true_unpert
    delta_inferred = stat_inferred_pert - stat_inferred_unpert
    
    mean_delta_inf = delta_inferred.mean(axis=0)
    best_delta_inf = delta_inferred[best_idx]
    
    # Remove ko_idx for plotting and metrics
    mask = np.ones(N_nodes, dtype=bool)
    mask[ko_idx] = False
    
    spearman_best, _ = spearmanr(delta_true[mask], best_delta_inf[mask])
    spearman_ens, _ = spearmanr(delta_true[mask], mean_delta_inf[mask])
    
    calib_metrics = evaluate_calibration(true_pert, stat_inferred_pert, ko_idx=ko_idx, alpha=0.05)
    kde_metrics = evaluate_multimodal_kde(true_pert, stat_inferred_pert, ko_idx=ko_idx)
    
    print("Generating 3-Panel Paper Figure...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    
    # Perfect match range
    min_val = min(delta_true[mask].min(), mean_delta_inf[mask].min(), best_delta_inf[mask].min()) - 0.2
    max_val = max(delta_true[mask].max(), mean_delta_inf[mask].max(), best_delta_inf[mask].max()) + 0.2
    
    # --- PANEL A: Single Best Network ---
    ax = axes[0]
    ax.scatter(
        delta_true[mask], best_delta_inf[mask],
        color='crimson', s=80, edgecolor='white', linewidth=1, label="MAP Prediction", zorder=10
    )
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7)
    
    ax.set_xlabel("True $\Delta$ Steady State", fontweight='bold')
    ax.set_ylabel("Single Best $\Delta$ Steady State", fontweight='bold')
    #ax.set_title(f"A. Point-Estimate Failure\n(Spearman $\\rho$ = {spearman_best:.2f})")
    ax.legend(loc="upper left")
    
    lower_bound = np.percentile(delta_inferred, 2.5, axis=0)
    upper_bound = np.percentile(delta_inferred, 97.5, axis=0)
    yerr = [mean_delta_inf - lower_bound, upper_bound - mean_delta_inf]
    
    ax.errorbar(
        delta_true[mask], mean_delta_inf[mask], yerr=[yerr[0][mask], yerr[1][mask]],
        fmt='o', color='dodgerblue', ecolor='lightgray', elinewidth=2.5, capsize=4,
        markersize=9, markeredgecolor='white', label="Ensemble Expected"
    )
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7)
    
    ax.set_xlabel("True $\Delta$ Steady State", fontweight='bold')
    ax.set_ylabel("Expected Inferred $\Delta$ Steady State", fontweight='bold')
    #ax.set_title(f"B. Ensemble Robustness\n(Spearman $\\rho$ = {spearman_ens:.2f}, Calib = {calib_metrics['fraction_within_ci']:.0%})")
    ax.legend(loc="upper left")
    
    # --- PANEL C: KDE Density for a divergent node ---
    ax = axes[1]
    
    target_node = 8 # Node 8 showed massive divergence between MAP and Ensemble
    
    sns.kdeplot(stat_inferred_pert[:, target_node], ax=ax, fill=True, color='mediumpurple', alpha=0.5, lw=2)
    
    # Mark true state
    ax.axvline(true_pert[target_node], color='forestgreen', linestyle='-', lw=3, label=f"True State")
    
    # Mark Single Best state
    ax.axvline(stat_inferred_pert[best_idx, target_node], color='crimson', linestyle='--', lw=2.5, label="Single Best (MAP)")
    
    # Mark ensemble mean
    ax.axvline(stat_inferred_pert[:, target_node].mean(), color='dodgerblue', linestyle=':', lw=2.5, label="Ensemble Mean")
    
    ax.set_xlabel(f"Perturbed Steady State (Node {target_node+1})", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')
    ax.set_ylim([0, 0.8])
    #ax.set_title("C. Resolving Uncertainty (Node 9)")
    ax.legend(loc="upper left")
    
    plt.tight_layout(pad=3.0)
    os.makedirs("papers_figures", exist_ok=True)
    plt.savefig("papers_figures/ensemble_advantage.png", dpi=300, bbox_inches='tight')
    print("Figure saved to papers_figures/ensemble_advantage.png")

if __name__ == "__main__":
    main()
