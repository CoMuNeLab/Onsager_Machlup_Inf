import numpy as np
import graph_tool.all as gt
import sys
sys.path.append('.')
from gene_expr import GeneExprBlockState
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.data_generation.network_generator import generate_scale_free_W

# Re-simulate the exact data to get the state (seed=123)
W_true = generate_scale_free_W(10, 1, seed=42)
s, _, _, _, _ = simulate_realistic_pairs(
    W_true, M=100, dt_sim=0.005, dt_sample=0.5, dt_meas=0.5, 
    D_base=0.02, mean_d=0.5, std_d=0.0, mean_V=2.0, std_V=0.0, 
    seed=123, t_window=(0.0, 3.0)
)

data = np.load('results/grn_ensemble_results.npz')
W_ensemble = data['W_ensemble']
theta_ensemble = data['theta_ensemble']

entropies = []
for i in range(100):
    g = gt.Graph(directed=True)
    g.add_vertex(10)
    W = W_ensemble[i]
    edges = np.argwhere(W != 0)
    for u, v in edges:
        g.add_edge(u, v)
        
    state = GeneExprBlockState(s, dt=0.5, g=g, state_args=dict(deg_corr=False), theta=theta_ensemble[i], use_em_likelihood=False)
    entropies.append(state.entropy())

best_idx = np.argmin(entropies)
print(f"Network {best_idx} has the lowest entropy: {entropies[best_idx]:.5f}")


N_nodes = 10
N_samples = 100
stat_inferred_pert = np.fromfile("results/stat_inferred_pert.bin", dtype=np.float64).reshape((N_nodes, N_samples), order='F').T
stat_inferred_unpert = np.fromfile("results/stat_inferred_unpert.bin", dtype=np.float64).reshape((N_nodes, N_samples), order='F').T
stat_true_pert_ens = np.fromfile("results/stat_true_pert.bin", dtype=np.float64).reshape((N_nodes, 100), order='F').T
stat_true_unpert_ens = np.fromfile("results/stat_true_unpert.bin", dtype=np.float64).reshape((N_nodes, 100), order='F').T

true_pert = stat_true_pert_ens.mean(axis=0)
true_unpert = stat_true_unpert_ens.mean(axis=0)

delta_true = true_pert - true_unpert
delta_best = stat_inferred_pert[best_idx] - stat_inferred_unpert[best_idx]
delta_ens = stat_inferred_pert.mean(axis=0) - stat_inferred_unpert.mean(axis=0)

from scipy.stats import spearmanr
print(f"Spearman (Lowest Entropy Best): {spearmanr(delta_true[1:], delta_best[1:])[0]:.3f}")
print(f"Spearman (Ensemble): {spearmanr(delta_true[1:], delta_ens[1:])[0]:.3f}")

for i in range(1, N_nodes):
    true_d = delta_true[i]
    best_d = delta_best[i]
    ens_d = delta_ens[i]
    # Error of best
    err_best = abs(best_d - true_d)
    err_ens = abs(ens_d - true_d)
    print(f"Node {i}: True={true_d:.3f}, Best={best_d:.3f} (Err {err_best:.3f}), Ens={ens_d:.3f} (Err {err_ens:.3f})")

