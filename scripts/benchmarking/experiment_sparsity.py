import os
import sys
import argparse
import numpy as np
import graph_tool.all as gt

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from gene_expr import GeneExprBlockState


def get_existing_trials(base_dir, N, M_pairs, dt_meas, mode):
    prefix = f"N_{N}_M_{M_pairs}_dt_{dt_meas}_mode_{mode}"
    existing = []
    if os.path.exists(base_dir):
        for entry in os.listdir(base_dir):
            if entry.startswith(prefix + "_trial_"):
                npz_file = os.path.join(base_dir, entry, "results.npz")
                if os.path.exists(npz_file):
                    try:
                        t_id = int(entry.split("_trial_")[-1])
                        existing.append(t_id)
                    except ValueError:
                        pass
    return sorted(existing)


def run_trial(N, M_pairs, dt_meas, dt_sample, dt_sim, mode, trial_id, output_base_dir, n_burnin=50, n_samples=100):
    config_name = f"N_{N}_M_{M_pairs}_dt_{dt_meas}_mode_{mode}_trial_{trial_id}"
    trial_dir = os.path.join(output_base_dir, config_name)
    npz_path = os.path.join(trial_dir, "results.npz")

    if os.path.exists(npz_path):
        print(f"Trial {trial_id} ({mode}) already exists at {npz_path}. Skipping.")
        return

    os.makedirs(trial_dir, exist_ok=True)

    # Identical seeds for matching trials
    seed_network = 42 + trial_id
    seed_data = 100 + trial_id

    # 1. Generate Ground Truth Network
    m = 1 if N <= 10 else 2
    W_true = generate_scale_free_W(N, m=m, seed=seed_network)
    W_true_binary = (W_true != 0).astype(int)

    # 2. Simulate Time-Series Pairs
    s_train, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
        W_true, M=M_pairs, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas,
        D_base=0.02, mean_d=0.5, std_d=0.0, mean_V=2.0, std_V=0.0,
        seed=seed_data, t_window=(0.0, 3.0)
    )

    # 3. Setup Initial Graph (~5% random edges to avoid empty-graph trap)
    g_init = gt.Graph(directed=True)
    g_init.add_vertex(N)
    num_initial_edges = max(1, int(N * N * 0.05))
    possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
    np.random.seed(seed_network)
    np.random.shuffle(possible_edges)
    for u, v in possible_edges[:num_initial_edges]:
        g_init.add_edge(u, v)

    initial_theta = [0.0, 0.0, np.log(0.05)]

    # 4. Bayesian MCMC Inference
    print(f"\n--- Running [{mode.upper()}] Mode | Trial {trial_id} (N={N}, M={M_pairs}, dt={dt_meas}) ---")
    state = GeneExprBlockState(
        s_train, dt=dt_meas, g=g_init.copy(),
        state_args=dict(deg_corr=False), theta=initial_theta.copy(), self_loops=False
    )
    state = run_burn_in(state, n_burnin=n_burnin, theta_mode=mode)
    W_ensemble, theta_ensemble = run_posterior_sampling(
        state, N, n_samples=n_samples, delta_k=1, theta_mode=mode
    )

    edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)

    # 5. Save Results Incrementally (like grid_search)
    np.savez_compressed(
        npz_path,
        W_true=W_true,
        W_true_binary=W_true_binary,
        W_ensemble=W_ensemble,
        theta_ensemble=theta_ensemble,
        edge_probabilities=edge_probabilities,
        weight_mean=weight_mean,
        weight_std=weight_std,
        N=N,
        M_pairs=M_pairs,
        dt_meas=dt_meas,
        dt_sample=dt_sample,
        mode=mode,
        trial_id=trial_id
    )
    print(f"Successfully saved trial to: {npz_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Global vs Local Sparsity/Overfitting Benchmark (Incremental)")
    parser.add_argument("--N", type=int, default=10, help="Number of nodes (default: 10)")
    parser.add_argument("--M", type=int, default=500, help="Number of sample pairs (default: 500)")
    parser.add_argument("--dt_meas", type=float, default=0.05, help="Measurement interval dt_meas (default: 0.05)")
    parser.add_argument("--dt_sample", type=float, default=0.05, help="Sampling interval dt_sample (default: 0.05)")
    parser.add_argument("--dt_sim", type=float, default=0.005, help="Internal simulation step (default: 0.005)")
    parser.add_argument("--reps", type=int, default=10, help="Number of new replicates to add (default: 10)")
    parser.add_argument("--mode", type=str, choices=["both", "global", "local"], default="both",
                        help="Inference mode to run: 'global', 'local', or 'both' (default: 'both')")
    parser.add_argument("--burnin", type=int, default=50, help="Burn-in sweeps (default: 50)")
    default_out = os.path.join(project_root, "results", "benchmarking", "regularization", "local_global")
    parser.add_argument("--output_dir", type=str, default=default_out,
                        help="Output directory (default: results/benchmarking/regularization/local_global)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    modes_to_run = ["global", "local"] if args.mode == "both" else [args.mode]

    # Find the next starting trial ID so new experiments sum up incrementally with previous ones
    existing_trials = set()
    for m in modes_to_run:
        existing_trials.update(get_existing_trials(args.output_dir, args.N, args.M, args.dt_meas, m))

    start_trial = max(existing_trials) + 1 if existing_trials else 0
    trials_to_run = list(range(start_trial, start_trial + args.reps))

    print(f"==================================================")
    print(f" Incremental Benchmark: Global vs Local Inference")
    print(f" Config: N={args.N}, M={args.M}, dt_meas={args.dt_meas}")
    print(f" Modes: {modes_to_run}")
    print(f" Existing completed trials: {sorted(list(existing_trials)) if existing_trials else 'None'}")
    print(f" Scheduled new trials: {trials_to_run} (Total new: {args.reps})")
    print(f" Output folder: {args.output_dir}")
    print(f"==================================================")

    for trial_id in trials_to_run:
        for mode in modes_to_run:
            run_trial(
                N=args.N,
                M_pairs=args.M,
                dt_meas=args.dt_meas,
                dt_sample=args.dt_sample,
                dt_sim=args.dt_sim,
                mode=mode,
                trial_id=trial_id,
                output_base_dir=args.output_dir,
                n_burnin=args.burnin,
                n_samples=args.samples
            )

    print("\nAll scheduled trials completed.")


if __name__ == "__main__":
    main()
