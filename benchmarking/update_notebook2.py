import json

nb_path = '/home/francesco/Universita/PhD/PROJECTS/InferenceBio/benchmarking/experiment_likelihood.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Find the loop cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'def calc_all_metrics' in ''.join(cell['source']):
        source = """import pandas as pd
from tqdm.notebook import tqdm
import time
import os

def calc_all_metrics(y_true_binary, y_true_continuous, y_scores, y_inferred_continuous=None):
    mask = ~np.eye(N, dtype=bool)
    y_true_b = y_true_binary[mask]
    y_s = y_scores[mask]
    
    fpr, tpr, _ = roc_curve(y_true_b, y_s)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true_b, y_s)
    pr_auc = auc(recall, precision)
    
    spearman_corr, _ = spearmanr(np.abs(y_true_continuous[mask]), np.abs(y_s))
    
    sign_acc = np.nan
    if y_inferred_continuous is not None:
        true_edges_mask = (y_true_b == 1)
        if np.any(true_edges_mask):
            true_signs = np.sign(y_true_continuous[mask][true_edges_mask])
            inf_signs = np.sign(y_inferred_continuous[mask][true_edges_mask])
            sign_acc = np.mean(true_signs == inf_signs)
    
    return fpr, tpr, roc_auc, precision, recall, pr_auc, spearman_corr, sign_acc

results_list = []

# Dynamically compute total intervals to keep constant across dt
t_span = t_window[1] - t_window[0]
base_dt = 0.5
total_intervals = M_train * (t_span / base_dt)

for seed_network in tqdm(seeds_network, desc="Network Seeds"):
    # Generate TRUE network
    W_true = generate_scale_free_W(N, m, seed=seed_network)
    W_true_binary = (W_true != 0).astype(int)
    theta_true = [np.log(mean_d), np.log(mean_V), np.log(D_base)]
    
    for dt_meas in tqdm(dt_vals, desc=f"dt values (seed {seed_network})", leave=False):
        dt_sample = dt_meas
        
        # Adjust M_train to keep the total number of measurement points constant
        intervals_per_traj = t_span / dt_meas
        M_dynamic = max(1, int(total_intervals / intervals_per_traj))
        
        # Simulate Training Data with M_dynamic
        s_train, _, _, _, _ = simulate_realistic_pairs(
            W_true, M=M_dynamic, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
            D_base=D_base, mean_d=mean_d, std_d=0.0, mean_V=mean_V, std_V=0.0, seed=100+seed_network, t_window=t_window
        )
        
        # --- Standard Likelihood ---
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)
        num_initial_edges = max(1, int(N * N * 0.05))
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.seed(42)
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:num_initial_edges]:
            g_init.add_edge(u, v)

        initial_theta = [0.0, 0.0, np.log(0.05)]
        
        # State without EM
        state_std = GeneExprBlockState(s_train, dt=dt_meas, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy(), use_em_likelihood=False)
        state_std = run_burn_in(state_std, n_burnin=100, theta_mode="global")
        W_ensemble_std, _ = run_posterior_sampling(state_std, N, n_samples=100, delta_k=1, theta_mode="global")
        edge_prob_std, weight_mean_std, _ = compute_ensemble_stats(W_ensemble_std)
        scores_std = edge_prob_std * np.abs(weight_mean_std)
        _, _, roc_auc_std, _, _, pr_auc_std, spearman_std, _ = calc_all_metrics(W_true_binary, W_true, scores_std, weight_mean_std)
        
        results_list.append({
            'Seed': seed_network,
            'dt': dt_meas,
            'M_train': M_dynamic,
            'Method': 'Standard',
            'AUROC': roc_auc_std,
            'AUPRC': pr_auc_std,
            'Spearman': spearman_std
        })
        
        # --- EM Likelihood ---
        state_em = GeneExprBlockState(s_train, dt=dt_meas, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy(), use_em_likelihood=True)
        state_em = run_burn_in(state_em, n_burnin=100, theta_mode="global")
        W_ensemble_em, _ = run_posterior_sampling(state_em, N, n_samples=100, delta_k=1, theta_mode="global")
        edge_prob_em, weight_mean_em, _ = compute_ensemble_stats(W_ensemble_em)
        scores_em = edge_prob_em * np.abs(weight_mean_em)
        _, _, roc_auc_em, _, _, pr_auc_em, spearman_em, _ = calc_all_metrics(W_true_binary, W_true, scores_em, weight_mean_em)
        
        results_list.append({
            'Seed': seed_network,
            'dt': dt_meas,
            'M_train': M_dynamic,
            'Method': 'EM',
            'AUROC': roc_auc_em,
            'AUPRC': pr_auc_em,
            'Spearman': spearman_em
        })

# Save results
os.makedirs("data", exist_ok=True)
df_results = pd.DataFrame(results_list)
df_results.to_csv("data/likelihood_benchmark_results.csv", index=False)
display(df_results.head())"""
        cell['source'] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]] if source else []
        cell['outputs'] = []  # Clear outputs so the notebook isn't cluttered
        break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
