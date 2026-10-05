import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import spearmanr
import graph_tool.all as gt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import plot_probabilistic_network, plot_inferred_vs_true_am, compute_reconstruction_metrics
from inference_pipeline.prediction.export_utils import export_ensemble_to_bin, export_true_to_bin
from gene_expr import GeneExprBlockState

def calc_auprc(y_true_binary, y_scores, N):
    mask = ~np.eye(N, dtype=bool)
    y_true_b = y_true_binary[mask]
    y_s = y_scores[mask]
    
    precision, recall, _ = precision_recall_curve(y_true_b, y_s)
    pr_auc = auc(recall, precision)
    return pr_auc

def calc_mae(W_true, W_inferred, N):
    mask = ~np.eye(N, dtype=bool)
    return np.mean(np.abs(W_true[mask] - W_inferred[mask]))

def calc_spearman(W_true, W_inferred, N):
    mask = ~np.eye(N, dtype=bool)
    return spearmanr(W_true[mask], W_inferred[mask])[0]

def run_experiment():
    np.random.seed(42)
    N = 10
    M_train = 500  # Rich dataset
    dt_sim = 0.005
    dt_sample = 0.05
    dt_meas = 0.05
    n_reps = 10
    
    results = {
        'Global': {'edges': [], 'auprc': [], 'mae': [], 'spearman': []},
        'Local': {'edges': [], 'auprc': [], 'mae': [], 'spearman': []}
    }
    
    for rep in range(n_reps):
        print(f"\n=======================")
        print(f"   Replicate {rep+1}/{n_reps}")
        print(f"=======================")
        
        seed = 42 + rep
        W_true = generate_scale_free_W(N, m=1, seed=seed)
        W_true_binary = (W_true != 0).astype(int)
        
        s_train, _, _, _, _ = simulate_realistic_pairs(
            W_true, M=M_train, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
            D_base=0.02, mean_d=0.5, std_d=0.0, mean_V=2.0, std_V=0.0, seed=100+rep, t_window=(0.0, 3.0)
        )
        
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)
        
        # Add ~5% density of random edges to jump-start the MCMC
        num_initial_edges = max(1, int(N * N * 0.05))
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.seed(seed)
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:num_initial_edges]:
            g_init.add_edge(u, v)
            
        initial_theta = [0.0, 0.0, np.log(0.05)]
        
        # --- GLOBAL MODE ---
        print("\n--- Running GLOBAL Mode ---")
        state_g = GeneExprBlockState(s_train, dt=dt_meas, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy())
        state_g = run_burn_in(state_g, n_burnin=50, theta_mode="global")
        W_ensemble_g, _ = run_posterior_sampling(state_g, N, n_samples=100, delta_k=1, theta_mode="global")
        
        edges_g = [np.sum(W != 0) for W in W_ensemble_g]
        results['Global']['edges'].extend(edges_g)
        
        ep_g, wm_g, _ = compute_ensemble_stats(W_ensemble_g)
        pr_g = calc_auprc(W_true_binary, ep_g * np.abs(wm_g), N)
        mae_g = calc_mae(W_true, ep_g * wm_g, N)
        spearman_g = calc_spearman(W_true, ep_g * wm_g, N)
        results['Global']['auprc'].append(pr_g)
        results['Global']['mae'].append(mae_g)
        results['Global']['spearman'].append(spearman_g)
        
        # --- LOCAL MODE ---
        print("\n--- Running LOCAL Mode ---")
        state_l = GeneExprBlockState(s_train, dt=dt_meas, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy())
        state_l = run_burn_in(state_l, n_burnin=50, theta_mode="local")
        W_ensemble_l, _ = run_posterior_sampling(state_l, N, n_samples=100, delta_k=1, theta_mode="local")
        
        edges_l = [np.sum(W != 0) for W in W_ensemble_l]
        results['Local']['edges'].extend(edges_l)
        
        ep_l, wm_l, _ = compute_ensemble_stats(W_ensemble_l)
        pr_l = calc_auprc(W_true_binary, ep_l * np.abs(wm_l), N)
        mae_l = calc_mae(W_true, ep_l * wm_l, N)
        spearman_l = calc_spearman(W_true, ep_l * wm_l, N)
        results['Local']['auprc'].append(pr_l)
        results['Local']['mae'].append(mae_l)
        results['Local']['spearman'].append(spearman_l)
        
    # Save raw results
    import pickle
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    results_path = os.path.join(plot_dir, 'sparsity_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
        
    # Plotting (Publication Quality)
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. Spearman Plot (Boxplot + Swarmplot for 5 reps)
    ax_spearman = axes[0]
    data_spearman = [results['Global']['spearman'], results['Local']['spearman']]
    sns.boxplot(data=data_spearman, ax=ax_spearman, palette=['#4C72B0', '#DD8452'], showfliers=False, width=0.5, boxprops={'alpha': 0.6})
    sns.stripplot(data=data_spearman, ax=ax_spearman, color='black', size=8, jitter=True, alpha=0.8)
    ax_spearman.set_xticks([0, 1])
    ax_spearman.set_xticklabels(['Global\n(Shared)', 'Local\n(Node-Specific)'])
    ax_spearman.set_ylabel('Spearman Correlation')
    ax_spearman.set_title('Inference Accuracy (Weights)', pad=15)
    sns.despine(ax=ax_spearman)
    
    # 2. AUPRC Plot (Boxplot + Swarmplot for 5 reps)
    ax_pr = axes[1]
    data_pr = [results['Global']['auprc'], results['Local']['auprc']]
    
    # We use a boxplot and overlay the 5 replicate points
    sns.boxplot(data=data_pr, ax=ax_pr, palette=['#4C72B0', '#DD8452'], showfliers=False, width=0.5, boxprops={'alpha': 0.6})
    sns.stripplot(data=data_pr, ax=ax_pr, color='black', size=8, jitter=True, alpha=0.8)
    
    ax_pr.set_xticks([0, 1])
    ax_pr.set_xticklabels(['Global\n(Shared)', 'Local\n(Node-Specific)'])
    ax_pr.set_ylabel('AUPRC (Structural Recovery)')
    ax_pr.set_title('Inference Accuracy', pad=15)
    ax_pr.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax_pr)
    
    plt.tight_layout()
    
    # Save as PNG and PDF for publications
    plot_path_png = os.path.join(plot_dir, 'sparsity_comparison_pub.png')
    plot_path_pdf = os.path.join(plot_dir, 'sparsity_comparison_pub.pdf')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {plot_path_png} and {plot_path_pdf}")
    
    # Print numerical summary
    print("\n--- Summary Statistics ---")
    print(f"Global Mode:")
    print(f"  Mean MAE: {np.mean(results['Global']['mae']):.4f} ± {np.std(results['Global']['mae']):.4f}")
    print(f"  Mean AUPRC: {np.mean(results['Global']['auprc']):.4f} ± {np.std(results['Global']['auprc']):.4f}")
    print(f"Local Mode:")
    print(f"  Mean MAE: {np.mean(results['Local']['mae']):.4f} ± {np.std(results['Local']['mae']):.4f}")
    print(f"  Mean AUPRC: {np.mean(results['Local']['auprc']):.4f} ± {np.std(results['Local']['auprc']):.4f}")

if __name__ == "__main__":
    run_experiment()
