import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure we can import the metrics and locate the results folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.append(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from perturbation_metrics import evaluate_shift_correlation, evaluate_calibration, evaluate_multimodal_kde
from scipy.stats import spearmanr

# --- GLOBAL PLOT CONFIGURATION ---
FONT_SCALE = 2.0          # Increased base scaling
TITLE_FONT_SIZE = 0      # Much larger titles
LABEL_FONT_SIZE = 25      # Larger axis labels
TICK_FONT_SIZE = 25       # Larger ticks
LEGEND_FONT_SIZE = 20     # Larger legends
MARKER_SIZE = 18          # Larger scatter points
LINE_WIDTH = 3            # Thicker lines
# ---------------------------------

def load_julia_binary(filepath, shape):
    """Loads a Julia-exported binary file (column-major) into a numpy array."""
    data = np.fromfile(filepath, dtype=np.float64)
    return data.reshape(shape, order='F').T

def main():
    print("Loading data...")
    N_nodes = 10
    bytes_per_float = 8
    
    pert_dir = os.path.join(PROJECT_ROOT, "results", "perturbation")
    if not os.path.exists(pert_dir):
        pert_dir = os.path.join(PROJECT_ROOT, "results")
        
    inferred_pert_path = os.path.join(pert_dir, "stat_inferred_pert.bin")
    inferred_unpert_path = os.path.join(pert_dir, "stat_inferred_unpert.bin")
    true_pert_path = os.path.join(pert_dir, "stat_true_pert.bin")
    true_unpert_path = os.path.join(pert_dir, "stat_true_unpert.bin")
    
    N_samples = int(os.path.getsize(inferred_pert_path) / (N_nodes * bytes_per_float))
    N_true_samples = int(os.path.getsize(true_pert_path) / (N_nodes * bytes_per_float))
    
    stat_inferred_pert = load_julia_binary(inferred_pert_path, (N_nodes, N_samples))
    stat_inferred_unpert = load_julia_binary(inferred_unpert_path, (N_nodes, N_samples))
    stat_true_pert_ens = load_julia_binary(true_pert_path, (N_nodes, N_true_samples))
    stat_true_unpert_ens = load_julia_binary(true_unpert_path, (N_nodes, N_true_samples))
    
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
    
    print("Generating 2-Panel Paper Figure...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update({
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.labelsize': LABEL_FONT_SIZE,
        'xtick.labelsize': TICK_FONT_SIZE,
        'ytick.labelsize': TICK_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE
    })
    
    # 2x1 layout requested by user
    fig, axes = plt.subplots(2, 1, figsize=(10, 14))
    
    # Perfect match range
    min_val = min(delta_true[mask].min(), mean_delta_inf[mask].min(), best_delta_inf[mask].min()) - 0.2
    max_val = max(delta_true[mask].max(), mean_delta_inf[mask].max(), best_delta_inf[mask].max()) + 0.2
    
    # --- PANEL A: Combined Scatter Plot ---
    ax = axes[0]
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=LINE_WIDTH, alpha=0.7, label="Perfect Prediction")
    
    # Plot MAP prediction
    ax.scatter(
        delta_true[mask], best_delta_inf[mask],
        color='crimson', s=MARKER_SIZE**2, edgecolor='white', linewidth=1.5, label=f"MAP Prediction ($\\rho$ = {spearman_best:.2f})", marker='^', zorder=3
    )
    
    # Plot Ensemble Expected with Error Bars
    lower_bound = np.percentile(delta_inferred, 2.5, axis=0)
    upper_bound = np.percentile(delta_inferred, 97.5, axis=0)
    yerr = [mean_delta_inf - lower_bound, upper_bound - mean_delta_inf]
    
    ax.errorbar(
        delta_true[mask], mean_delta_inf[mask], yerr=[yerr[0][mask], yerr[1][mask]],
        fmt='o', color='dodgerblue', ecolor='dodgerblue', elinewidth=LINE_WIDTH, capsize=6, capthick=LINE_WIDTH-1,
        markersize=MARKER_SIZE, markeredgecolor='white', label=f"Ensemble Expected ($\\rho$ = {spearman_ens:.2f})", zorder=2
    )
    
    ax.set_xlabel(r"True $\Delta$ Steady State")
    ax.set_ylabel(r"Predicted $\Delta$ Steady State")
    ax.set_xlim([min_val, max_val])
    #ax.set_ylim([min_val, max_val+0.1])
    ax.set_title("A. Systemic Shift Prediction: MAP vs Ensemble")
    ax.legend(loc="upper left")
    
    # --- PANEL B: KDE Density for a divergent node (Node 9) ---
    ax = axes[1]
    target_node = 8 # Node 8 (Node 9 in 1-indexed) showed massive divergence
    
    sns.kdeplot(stat_inferred_pert[:, target_node], ax=ax, fill=True, color='lightblue', alpha=0.5, lw=2, label="Ensemble KDE")
    
    # Mark true state
    ax.axvline(true_pert[target_node], color='darkgray', linestyle='-', lw=LINE_WIDTH+1, label=f"True State")
    
    # Mark Single Best state
    ax.axvline(stat_inferred_pert[best_idx, target_node], color='crimson', linestyle='--', lw=LINE_WIDTH, label="Single Best (MAP)")
    
    # Mark ensemble mean
    ax.axvline(stat_inferred_pert[:, target_node].mean(), color='dodgerblue', linestyle=':', lw=LINE_WIDTH, label="Ensemble Mean")
    
    ax.set_xlabel(f"Perturbed Steady State (Node {target_node+1})")
    ax.set_ylabel("Probability Density")
    ax.set_title("B. Resolving Uncertainty with Ensembles (Node 9)")
    ax.set_xlim([0.5, 6.0])
    #ax.set_ylim([min_val, max_val])
    ax.legend(loc="upper right")
    
    plt.tight_layout(pad=3.0)
    out_dir = os.path.join(PROJECT_ROOT, "papers_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ensemble_advantage.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    
    img_dir = os.path.join(PROJECT_ROOT, "images", "analysis")
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, "ensemble_advantage.png"), dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path} and {os.path.join(img_dir, 'ensemble_advantage.png')}")

if __name__ == "__main__":
    main()
