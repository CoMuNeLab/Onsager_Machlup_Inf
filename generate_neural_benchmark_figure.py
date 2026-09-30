import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import FancyArrowPatch
import graph_tool.all as gt
from scipy.special import expit
import os
import sys

# Append path to import inference modules
sys.path.append(os.path.abspath('.'))
from gene_expr import GeneExprBlockState
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats

# Set publication quality aesthetics
sns.set_theme(style="white", context="paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

output_dir = '/home/francesco/Universita/PhD/PROJECTS/InferenceBio/papers_figures/'
os.makedirs(output_dir, exist_ok=True)

def generate_data_and_infer():
    np.random.seed(42)
    N = 10
    p_edge = 0.15

    G_true = nx.erdos_renyi_graph(n=N, p=p_edge, directed=True, seed=42)
    W_true = nx.to_numpy_array(G_true)
    signs = np.random.choice([-1.0, 1.0], size=(N, N))
    true_weights = np.random.uniform(0.5, 1.5, size=(N, N))
    W_true = W_true * signs * true_weights

    tau = 2.0         
    I = -1.5          
    sigma = 0.05      
    dt_sim = 0.01
    steps_per_trial = 200  
    num_trials = 30        
    subsample = 10         

    print(f"Simulating Wilson-Cowan Neural Network (N={N})...")
    pairs = []
    trajectories = []

    for trial in range(num_trials):
        x = np.random.uniform(0.1, 0.9, size=N)
        trial_traj = []
        for step in range(steps_per_trial):
            net_input = W_true @ x + I
            activation = expit(net_input)
            dx = (-x / tau + (1 - x) * activation) * dt_sim
            noise = sigma * np.sqrt(dt_sim) * np.random.randn(N)
            x = x + dx + noise
            x = np.clip(x, 0.01, 0.99)
            if step % subsample == 0:
                trial_traj.append(x.copy())
        
        for i in range(len(trial_traj) - 1):
            pairs.append(np.column_stack([trial_traj[i], trial_traj[i+1]]))
        trajectories.append(np.array(trial_traj))

    dt_infer = dt_sim * subsample
    print(f"Generated {len(pairs)} transition pairs with dt={dt_infer}")

    g_init = gt.Graph(directed=True)
    g_init.add_vertex(N)
    g_init.add_edge(0, 1)

    initial_theta = [np.log(1/tau), np.log(1.0), np.log(0.01)]

    print("Initializing Inference Engine...")
    state = GeneExprBlockState(pairs, dt=dt_infer, g=g_init, theta=initial_theta, self_loops=False)
    
    print("Running Burn-in (100 sweeps)...")
    state = run_burn_in(state, n_burnin=100, seed=42)
    
    print("Running Posterior Sampling (100 samples)...")
    W_ens, _ = run_posterior_sampling(state, N, n_samples=100, seed=42)
    
    edge_probs, weight_mean, weight_std = compute_ensemble_stats(W_ens)
    expected_weights = edge_probs * weight_mean
    
    return W_true, edge_probs, expected_weights, weight_std, trajectories

def plot_figure(W_true, edge_probs, expected_weights, weight_std, trajectories):
    fig = plt.figure(figsize=(26, 4.5))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 6, width_ratios=[1.3, 1, 1, 1, 1.1, 1.1], wspace=0.4)
    
    # ------------------------------------------------------------------
    # Panel A: Trajectories with Title/Equations
    # ------------------------------------------------------------------
    ax_traj = fig.add_subplot(gs[0, 0])
    concat_traj = np.concatenate(trajectories[:2], axis=0)
    ax_traj.plot(concat_traj, alpha=0.7)
    ax_traj.axvline(len(trajectories[0]), color='black', linestyle='--', alpha=0.5)
    
    eq_text = r"$\tau \frac{dx_i}{dt} = -x_i + (1-x_i)\sigma\left(\sum_{j} W_{ij}x_j + I\right)$"
    ax_traj.set_title(f"Wilson-Cowan Rate Model Transients\n{eq_text}", fontsize=13, pad=15)
    ax_traj.set_xlabel("Time Step")
    ax_traj.set_ylabel("Firing Rate $x_i(t)$")
    ax_traj.set_xlim(0, len(concat_traj))
    ax_traj.set_ylim(0, 1.0)

    # ------------------------------------------------------------------
    # Panel B: Heatmaps
    # ------------------------------------------------------------------
    ax_true = fig.add_subplot(gs[0, 1])
    vmax = np.abs(W_true).max()
    sns.heatmap(W_true, ax=ax_true, cmap="vlag", center=0, vmin=-vmax, vmax=vmax, 
                cbar_kws={'label': 'Weight', 'shrink': 0.8}, square=True)
    ax_true.set_title("True Target Topology\n($W_{true}$)")
    ax_true.set_xlabel("Source Node")
    ax_true.set_ylabel("Target Node")
    
    ax_prob = fig.add_subplot(gs[0, 2])
    sns.heatmap(edge_probs, ax=ax_prob, cmap="Reds", vmin=0, vmax=1.0,
                cbar_kws={'label': 'Probability', 'shrink': 0.8}, square=True)
    ax_prob.set_title("Inferred Edge Probs\n($P_{ij}$)")
    ax_prob.set_xlabel("Source Node")
    
    ax_exp = fig.add_subplot(gs[0, 3])
    vmax_exp = np.abs(expected_weights).max()
    sns.heatmap(expected_weights, ax=ax_exp, cmap="vlag", center=0, vmin=-vmax_exp, vmax=vmax_exp,
                cbar_kws={'label': 'Expected Weight', 'shrink': 0.8}, square=True)
    ax_exp.set_title("Inferred Expected Weights\n($\\mathbb{E}[W_{ij}]$)")
    ax_exp.set_xlabel("Source Node")

    # ------------------------------------------------------------------
    # Panel C: ROC Curve & Weight Correlation Scatter
    # ------------------------------------------------------------------
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    y_true_binary = (W_true[mask] != 0).astype(int)
    y_scores = edge_probs[mask]
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    ax_roc = fig.add_subplot(gs[0, 4])
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Structural Recovery\n(ROC Curve)')
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle='--', alpha=0.5)

    ax_corr = fig.add_subplot(gs[0, 5])
    x_vals = W_true[mask]
    y_vals = expected_weights[mask]
    y_errs = weight_std[mask]
    
    pr, _ = pearsonr(x_vals, y_vals)
    sr, _ = spearmanr(x_vals, y_vals)
    
    true_edge_mask = (x_vals != 0)
    ax_corr.errorbar(x_vals[~true_edge_mask], y_vals[~true_edge_mask], yerr=y_errs[~true_edge_mask], fmt='o', color='gray', alpha=0.5, label='True Null Edge')
    ax_corr.errorbar(x_vals[true_edge_mask], y_vals[true_edge_mask], yerr=y_errs[true_edge_mask], fmt='o', color='purple', alpha=0.8, label='True Edge')
    
    m, b = np.polyfit(x_vals, y_vals, 1)
    ax_corr.plot(x_vals, m*x_vals + b, color='black', linestyle='--', alpha=0.6, label='Linear Fit')
    
    ax_corr.set_title(f"Weight Recovery\nPearson $\\rho$={pr:.2f}, Spearman $r$={sr:.2f}")
    ax_corr.set_xlabel("True Weights ($W_{true}$)")
    ax_corr.set_ylabel("Expected Weights ($\\mathbb{E}[W_{ij}]$)")
    ax_corr.legend(loc='lower right', fontsize=9)
    ax_corr.grid(True, linestyle='--', alpha=0.5)

    # Add Panel Labels
    fig.text(0.01, 0.95, "a", fontsize=24, fontweight='bold', va='top')
    fig.text(0.24, 0.95, "b", fontsize=24, fontweight='bold', va='top')
    fig.text(0.70, 0.95, "c", fontsize=24, fontweight='bold', va='top')

    plt.tight_layout(rect=[0.02, 0.0, 1.0, 1.0])
    
    out_path = os.path.join(output_dir, 'neural_benchmark_figure.pdf')
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    W_true, edge_probs, expected_weights, weight_std, trajectories = generate_data_and_infer()
    plot_figure(W_true, edge_probs, expected_weights, weight_std, trajectories)
