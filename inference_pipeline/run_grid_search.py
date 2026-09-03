import os
import sys
import csv
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import multiprocessing

# Restrict graph-tool OpenMP threads to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"

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
        initial_theta = [0.0, 0.0, np.log(0.05)]
        
        state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=True), theta=initial_theta)
        
        # 50 burn in, 100 sampling (from user request)
        state = run_burn_in(state, n_burnin=50, optimize_params=True)
        W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, optimize_params=True)
        
        edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
        
        # 3. Save NPZ
        npz_path = os.path.join(output_dir, "results.npz")
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
        return row
        
    except Exception as e:
        return {
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
    grid_M_pairs = [100, 200, 500, 1000, 2000]
    grid_t_window = [(0.0, 3.0), (10.0, 15.0)]
    grid_N = [10, 20, 50, 100, 200]
    
    # Check if we are doing a dry run
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("Running DRY RUN (subset of grid)")
        grid_dt_meas = [0.05]
        grid_M_pairs = [100]
        grid_t_window = [(0.0, 3.0)]
        grid_N = [10, 20]
        
    all_configs = list(itertools.product(grid_dt_meas, grid_M_pairs, grid_t_window, grid_N))
    print(f"Total configurations to run: {len(all_configs)}")
    
    os.makedirs("grid_results", exist_ok=True)
    csv_file = os.path.join("grid_results", "performance_metrics.csv")
    
    fieldnames = [
        "dt_meas", "M_pairs", "t_window_start", "t_window_end", "N",
        "TP", "FP", "FN", "TN", "TPR (Recall)", "FPR", "Precision", "F1-Score", "SHD",
        "status", "error"
    ]
    
    # Initialize CSV with header
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # We leave 1 core free for responsiveness
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(run_single_config, *config): config for config in all_configs}
        
        # Process them as they complete
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            config = futures[future]
            try:
                result_row = future.result()
                # Append to CSV sequentially to avoid race conditions
                with open(csv_file, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    # Fill missing metric keys if failed
                    for key in fieldnames:
                        if key not in result_row:
                            result_row[key] = None
                            
                    writer.writerow(result_row)
                
                print(f"[{completed_count}/{len(all_configs)}] Config {config} completed. Status: {result_row['status']}")
            except Exception as e:
                print(f"[{completed_count}/{len(all_configs)}] Config {config} raised an exception: {e}")

    total_time = time.time() - start_time
    print(f"Grid search finished in {total_time:.2f} seconds.")

