import os
import sys
import numpy as np
import pandas as pd
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import graph_tool.all as gt
from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import spearmanr

def calc_all_metrics(y_true_binary, y_true_continuous, y_scores, y_inferred_continuous=None):
    N = y_true_binary.shape[0]
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
from gene_expr import GeneExprBlockState

def main():
    parser = argparse.ArgumentParser(description="Run theoretical cases for EM vs Standard likelihood")
    parser.add_argument("--mean_d", type=float, default=0.5, help="Mean degradation rate")
    parser.add_argument("--D_base", type=float, default=0.02, help="Base noise level")
    parser.add_argument("--mean_V", type=float, default=2.0, help="Mean production rate")
    parser.add_argument("--dt", type=float, default=1.0, help="Sampling interval")
    parser.add_argument("--tag", type=str, default="test", help="Output file tag")
    args = parser.parse_args()

    N = 10
    m = 1
    seeds_network = [466, 860] # Use a couple known seeds to be fast
    
    dt_sim = 0.005
    dt_meas = args.dt
    M_train = 100
    t_window = (0.0, 10.0)

    results_list = []

    print(f"Running experiments with mean_d={args.mean_d}, D_base={args.D_base}, mean_V={args.mean_V}, dt={dt_meas}")

    for seed_network in seeds_network:
        W_true = generate_scale_free_W(N, m, seed=seed_network)
        W_true_binary = (W_true != 0).astype(int)
        
        # Simulate Data
        s_train, _, _, _, _ = simulate_realistic_pairs(
            W_true, M=M_train, dt_sim=dt_sim, dt_sample=dt_meas, dt_meas=dt_meas, 
            D_base=args.D_base, mean_d=args.mean_d, std_d=0.0, mean_V=args.mean_V, std_V=0.0, seed=100+seed_network, t_window=t_window
        )
        
        # Initial Graph
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.seed(42)
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:max(1, int(N*N*0.05))]:
            g_init.add_edge(u, v)
            
        initial_theta = [np.log(args.mean_d), np.log(args.mean_V), np.log(args.D_base)]

        # --- Standard Method ---
        try:
            state_std = GeneExprBlockState(s_train, dt=dt_meas, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy(), use_em_likelihood=False)
            state_std = run_burn_in(state_std, n_burnin=50, theta_mode="global")
            W_ensemble_std, _ = run_posterior_sampling(state_std, N, n_samples=50, delta_k=1, theta_mode="global")
            edge_prob_std, weight_mean_std, _ = compute_ensemble_stats(W_ensemble_std)
            scores_std = edge_prob_std * np.abs(weight_mean_std)
            _, _, roc_auc_std, _, _, pr_auc_std, _, _ = calc_all_metrics(W_true_binary, W_true, scores_std, weight_mean_std)
        except Exception as e:
            print(f"Standard failed for seed {seed_network}: {e}")
            roc_auc_std, pr_auc_std = np.nan, np.nan

        # --- EM Method ---
        try:
            state_em = GeneExprBlockState(s_train, dt=dt_meas, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy(), use_em_likelihood=True)
            state_em = run_burn_in(state_em, n_burnin=50, theta_mode="global")
            W_ensemble_em, _ = run_posterior_sampling(state_em, N, n_samples=50, delta_k=1, theta_mode="global")
            edge_prob_em, weight_mean_em, _ = compute_ensemble_stats(W_ensemble_em)
            scores_em = edge_prob_em * np.abs(weight_mean_em)
            _, _, roc_auc_em, _, _, pr_auc_em, _, _ = calc_all_metrics(W_true_binary, W_true, scores_em, weight_mean_em)
        except Exception as e:
            print(f"EM failed for seed {seed_network}: {e}")
            roc_auc_em, pr_auc_em = np.nan, np.nan

        results_list.append({
            'Seed': seed_network, 'dt': dt_meas, 'mean_d': args.mean_d, 'D_base': args.D_base, 'mean_V': args.mean_V,
            'Method': 'Standard', 'AUROC': roc_auc_std, 'AUPRC': pr_auc_std
        })
        results_list.append({
            'Seed': seed_network, 'dt': dt_meas, 'mean_d': args.mean_d, 'D_base': args.D_base, 'mean_V': args.mean_V,
            'Method': 'EM', 'AUROC': roc_auc_em, 'AUPRC': pr_auc_em
        })

    df = pd.DataFrame(results_list)
    out_file = f"results_{args.tag}.csv"
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")
    print(df.groupby('Method')[['AUROC', 'AUPRC']].mean())

if __name__ == "__main__":
    main()
