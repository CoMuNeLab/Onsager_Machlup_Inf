import os
import sys
import csv
import itertools
import numpy as np
import time

import graph_tool.all as gt

sys.path.append(os.path.abspath('..'))
from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import compute_reconstruction_metrics
from gene_expr import GeneExprBlockState

def run_single_config(dt_meas, M_pairs, t_window, N):
    try:
        # Fixed hyperparameters
        m = 1
        seed_network = 42
        dt_sim = 0.01
        dt_sample = 0.5
        mean_d = 0.5
        mean_V = 2.0
        D_base = 0.02
        
        # Output directory
        config_name = f"N_{N}_M_{M_pairs}_dt_{dt_meas}_twnd_{t_window[0]}_{t_window[1]}"
        output_dir = os.path.join("grid_results", config_name)
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, "results.npz")
        
        # Checkpoint: skip if already processed
        if os.path.exists(npz_path):
            return "skipped", None
        
        # 1. Generate Data
        W_true = generate_scale_free_W(N, m, seed=seed_network)
        theta_true = [np.log(mean_d), np.log(mean_V), np.log(D_base)]
        
        s, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
            W_true, M=M_pairs, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
            D_base=D_base, mean_d=mean_d, std_d=0.0, mean_V=mean_V, std_V=0.0, 
            seed=123, t_window=t_window
        )
        
        # 2. Bayesian Inference
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)
        
        # Add ~5% density of random edges to jump-start the MCMC and avoid the empty graph trap
        num_initial_edges = max(1, int(N * N * 0.05))
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:num_initial_edges]:
            g_init.add_edge(u, v)
            
        initial_theta = [0.0, 0.0, np.log(0.05)]
        
        state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=True), theta=initial_theta)
        
        # 50 burn in, 100 sampling
        state = run_burn_in(state, n_burnin=50, optimize_params=True)
        W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, optimize_params=True)
        
        edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
        
        # 3. Save NPZ
        np.savez_compressed(
            npz_path, 
            W_ensemble=W_ensemble, 
            theta_ensemble=theta_ensemble,
            edge_probabilities=edge_probabilities,
            weight_mean=weight_mean,
            weight_std=weight_std
        )
        
        # 4. Compute Metrics
        metrics = compute_reconstruction_metrics(W_true, W_ensemble[-1])
        
        # Return summary for CSV
        row = {
            "dt_meas": dt_meas,
            "M_pairs": M_pairs,
            "t_window_start": t_window[0],
            "t_window_end": t_window[1],
            "N": N,
            **metrics,
            "status": "success",
            "error": ""
        }
        return "success", row
        
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return "failed", {
            "dt_meas": dt_meas,
            "M_pairs": M_pairs,
            "t_window_start": t_window[0],
            "t_window_end": t_window[1],
            "N": N,
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    # Define Parameter Grid
    grid_dt_meas = [0.01, 0.05, 0.1, 0.5]
    grid_M_pairs = [100, 200, 500, 1000]
    grid_t_window = [(0.0, 3.0), (10.0, 15.0)]
    grid_N = [10, 20, 50, 100]
    
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Running DRY RUN (subset of grid)")
        grid_dt_meas = [0.05]
        grid_M_pairs = [100]
        grid_t_window = [(0.0, 3.0)]
        grid_N = [10, 20]
        
    all_configs = list(itertools.product(grid_dt_meas, grid_M_pairs, grid_t_window, grid_N))
    total_configs = len(all_configs)
    print(f"Total configurations to run: {total_configs}")
    
    os.makedirs("grid_results", exist_ok=True)
    csv_file = os.path.join("grid_results", "performance_metrics.csv")
    
    fieldnames = [
        "dt_meas", "M_pairs", "t_window_start", "t_window_end", "N",
        "TP", "FP", "FN", "TN", "TPR (Recall)", "FPR", "Precision", "F1-Score", "SHD",
        "status", "error"
    ]
    
    # Initialize CSV with header if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    start_time = time.time()
    
    try:
        for idx, config in enumerate(all_configs):
            dt_meas, M_pairs, t_window, N = config
            pct = (idx / total_configs) * 100
            print(f"\n[{idx+1}/{total_configs}] ({pct:.1f}%) Processing N={N}, M={M_pairs}, dt={dt_meas}, twnd={t_window} ...")
            
            status, result_row = run_single_config(*config)
            
            if status == "skipped":
                print(f"  -> SKIPPED (Already computed)")
            else:
                print(f"  -> {status.upper()}")
                if result_row:
                    with open(csv_file, mode="a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        for key in fieldnames:
                            if key not in result_row:
                                result_row[key] = None
                        writer.writerow(result_row)
    except KeyboardInterrupt:
        print("\n\nGrid search INTERRUPTED by user (Ctrl+C). Progress saved safely.")
        sys.exit(0)

    total_time = time.time() - start_time
    print(f"\nGrid search finished in {total_time:.2f} seconds.")
