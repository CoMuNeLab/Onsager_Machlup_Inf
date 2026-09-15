import numpy as np

# Load data
N_nodes = 10
N_samples = 100
stat_inferred_pert = np.fromfile("results/stat_inferred_pert.bin", dtype=np.float64).reshape((N_nodes, N_samples), order='F').T
stat_inferred_unpert = np.fromfile("results/stat_inferred_unpert.bin", dtype=np.float64).reshape((N_nodes, N_samples), order='F').T
stat_true_pert_ens = np.fromfile("results/stat_true_pert.bin", dtype=np.float64).reshape((N_nodes, 100), order='F').T
stat_true_unpert_ens = np.fromfile("results/stat_true_unpert.bin", dtype=np.float64).reshape((N_nodes, 100), order='F').T

true_pert = stat_true_pert_ens.mean(axis=0)
true_unpert = stat_true_unpert_ens.mean(axis=0)

# Find the single network that best fits the WILD-TYPE state
errors_wt = np.sum((stat_inferred_unpert - true_unpert)**2, axis=1)
best_wt_idx = np.argmin(errors_wt)
print(f"Network {best_wt_idx} has the lowest wild-type error: {errors_wt[best_wt_idx]:.5f}")

# How does this 'best' network perform on perturbation?
delta_true = true_pert - true_unpert
delta_best = stat_inferred_pert[best_wt_idx] - stat_inferred_unpert[best_wt_idx]

from scipy.stats import spearmanr
print(f"Spearman (Best): {spearmanr(delta_true[1:], delta_best[1:])[0]:.3f}")

# How does ensemble perform?
delta_ens = stat_inferred_pert.mean(axis=0) - stat_inferred_unpert.mean(axis=0)
print(f"Spearman (Ensemble): {spearmanr(delta_true[1:], delta_ens[1:])[0]:.3f}")

# Find a node where 'best' fails but ensemble gets it right
for i in range(1, N_nodes):
    true_d = delta_true[i]
    best_d = delta_best[i]
    ens_d = delta_ens[i]
    # Error of best
    err_best = abs(best_d - true_d)
    err_ens = abs(ens_d - true_d)
    print(f"Node {i}: True={true_d:.3f}, Best={best_d:.3f} (Err {err_best:.3f}), Ens={ens_d:.3f} (Err {err_ens:.3f})")

