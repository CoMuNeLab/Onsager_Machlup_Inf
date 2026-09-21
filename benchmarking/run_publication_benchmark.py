import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
import time
from itertools import product

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import graph_tool.all as gt
from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def calc_metrics(y_true_binary, y_scores):
    N = y_true_binary.shape[0]
    mask = ~np.eye(N, dtype=bool)
    y_true_b = y_true_binary[mask]
    y_s = y_scores[mask]
    
    if np.sum(y_true_b) == 0 or np.sum(y_true_b) == len(y_true_b):
        return np.nan, np.nan
        
    fpr, tpr, _ = roc_curve(y_true_b, y_s)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true_b, y_s)
    pr_auc = auc(recall, precision)
    
    return roc_auc, pr_auc

def run_single_job(args):
    seed, dt, mean_d, D_base, regime_name = args
    
    N = 10
    m = 1
    mean_V = 2.0
    dt_sim = 0.005
    M_train = 100
    t_window = (0.0, 10.0)
    
    results = []
    
    try:
        # Generate exactly identical network and data
        W_true = generate_scale_free_W(N, m, seed=seed)
        W_true_binary = (W_true != 0).astype(int)
        
        s_train, _, _, _, _ = simulate_realistic_pairs(
            W_true, M=M_train, dt_sim=dt_sim, dt_sample=dt, dt_meas=dt, 
            D_base=D_base, mean_d=mean_d, std_d=0.0, mean_V=mean_V, std_V=0.0, 
            seed=100+seed, t_window=t_window
        )
        
        # Initial Graph
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.seed(42)
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:max(1, int(N*N*0.05))]:
            g_init.add_edge(u, v)
            
        initial_theta = [np.log(mean_d), np.log(mean_V), np.log(D_base)]
        
        # We need to import GeneExprBlockState here if multiprocessing causes issues, but usually it's fine
        from gene_expr import GeneExprBlockState
        
        # --- Standard Method ---
        try:
            state_std = GeneExprBlockState(s_train, dt=dt, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy(), use_em_likelihood=False)
            state_std = run_burn_in(state_std, n_burnin=50, theta_mode="global")
            W_ensemble_std, _ = run_posterior_sampling(state_std, N, n_samples=50, delta_k=1, theta_mode="global")
            edge_prob_std, weight_mean_std, _ = compute_ensemble_stats(W_ensemble_std)
            scores_std = edge_prob_std * np.abs(weight_mean_std)
            roc_auc_std, pr_auc_std = calc_metrics(W_true_binary, scores_std)
        except Exception as e:
            print(f"[{regime_name} dt={dt} seed={seed}] Standard failed: {e}")
            roc_auc_std, pr_auc_std = np.nan, np.nan
            
        # --- EM Method ---
        try:
            state_em = GeneExprBlockState(s_train, dt=dt, g=g_init.copy(), state_args=dict(deg_corr=False), theta=initial_theta.copy(), use_em_likelihood=True)
            state_em = run_burn_in(state_em, n_burnin=50, theta_mode="global")
            W_ensemble_em, _ = run_posterior_sampling(state_em, N, n_samples=50, delta_k=1, theta_mode="global")
            edge_prob_em, weight_mean_em, _ = compute_ensemble_stats(W_ensemble_em)
            scores_em = edge_prob_em * np.abs(weight_mean_em)
            roc_auc_em, pr_auc_em = calc_metrics(W_true_binary, scores_em)
        except Exception as e:
            print(f"[{regime_name} dt={dt} seed={seed}] EM failed: {e}")
            roc_auc_em, pr_auc_em = np.nan, np.nan

        results.append({
            'Regime': regime_name, 'Seed': seed, 'dt': dt, 'mean_d': mean_d, 'D_base': D_base,
            'Method': 'Standard', 'AUROC': roc_auc_std, 'AUPRC': pr_auc_std
        })
        results.append({
            'Regime': regime_name, 'Seed': seed, 'dt': dt, 'mean_d': mean_d, 'D_base': D_base,
            'Method': 'EM', 'AUROC': roc_auc_em, 'AUPRC': pr_auc_em
        })
        
        return results
    except Exception as e:
        print(f"Job failed completely: {e}")
        return []

def main():
    regimes = [
        ("Baseline", 0.5, 0.02),
        ("Stiff", 2.0, 0.02),
        ("High_Noise", 0.5, 0.2)
    ]
    dt_vals = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]
    seeds = list(range(1, 11))  # 10 seeds
    
    jobs = []
    for (regime_name, mean_d, D_base) in regimes:
        for dt in dt_vals:
            for seed in seeds:
                jobs.append((seed, dt, mean_d, D_base, regime_name))
                
    print(f"Total jobs to run: {len(jobs)}")
    
    out_file = "data/publication_likelihood_benchmark.csv"
    os.makedirs("data", exist_ok=True)
    
    # We will write incrementally to not lose data
    if os.path.exists(out_file):
        os.remove(out_file)
        
    all_results = []
    
    # Use pool to run in parallel
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Running on {num_cores} cores...")
    
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for i, res_list in enumerate(pool.imap_unordered(run_single_job, jobs)):
            if res_list:
                all_results.extend(res_list)
                # Save incrementally
                df = pd.DataFrame(all_results)
                df.to_csv(out_file, index=False)
            
            if (i+1) % 10 == 0:
                print(f"Progress: {i+1}/{len(jobs)} jobs completed. Elapsed: {time.time()-start_time:.1f}s")
                
    print(f"Finished completely! Data saved to {out_file}")

if __name__ == "__main__":
    main()
