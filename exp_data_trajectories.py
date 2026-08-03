#!/usr/bin/env python3
"""
Experimental Investigation of Input Data & Trajectory Properties on GRN Inference Accuracy
using graph-tool C++ Extension (`GeneExprBlockState`).

Experiments:
  1. Measurement Interval (`dt_meas`): T_total=100.0, dt=0.01, dt_meas in [0.05, 0.1, 0.2, 0.5, 1.0]
  2. Trajectory Length / Sample Size (`M`): dt_meas=0.1, M in [50, 100, 200, 500, 1000]
  3. Noise Intensity (`noise_std`): T_total=100.0, dt=0.01, dt_meas=0.1, noise_std in [0.05, 0.15, 0.30, 0.50]
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add project root to sys.path
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from run_onsager_graph_tool_pipeline import (
    get_benchmark_10gene_network,
    simulate_grn_sde_trajectory,
    compute_reconstruction_metrics,
    sigmoid
)

# Plotting style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})

def compute_physical_diagnostics(s, W_true, dt_meas, d=0.5, V=2.0, noise_std=0.15):
    """
    Computes physical & dynamical diagnostics for trajectory s (shape N x M):
      - mean_var_x: Mean expression variance across time
      - mean_var_v: Mean finite-difference velocity variance across time
      - empirical_snr: Ratio of deterministic drift variance to Brownian noise variance
    """
    N, M = s.shape
    mean_var_x = float(np.mean(np.var(s, axis=1)))
    
    if M > 1:
        vel = (s[:, 1:] - s[:, :-1]) / dt_meas
        mean_var_v = float(np.mean(np.var(vel, axis=1)))
    else:
        mean_var_v = 0.0

    # Compute deterministic drift at each measured point
    # dx/dt = -d * x + V * sigmoid(W_true @ x)
    drifts = np.zeros_like(s)
    for t in range(M):
        x = s[:, t]
        prod = sigmoid(W_true @ x)
        drifts[:, t] = -d * x + V * prod

    var_drift = float(np.mean(np.var(drifts, axis=1)))
    var_noise_rate = float((noise_std ** 2) / dt_meas)
    empirical_snr = float(var_drift / var_noise_rate) if var_noise_rate > 0 else 0.0

    return {
        "mean_var_x": mean_var_x,
        "mean_var_v": mean_var_v,
        "var_drift": var_drift,
        "var_noise_rate": var_noise_rate,
        "empirical_snr": empirical_snr
    }

def run_single_experiment(W_true, s, dt_meas, noise_std=0.15, n_sweeps=30, burn_in=10, label=""):
    """
    Runs MCMC inference on trajectory tensor s (N x M) with measurement step dt_meas.
    Computes AUROC, AUPRC, and reconstruction metrics (at P_edge >= 0.5 and best_graph).
    """
    N, M = s.shape
    start_t = time.time()
    state = GeneExprBlockState(s, dt=dt_meas)
    
    entropy_history = [state.entropy()]
    edge_counts = np.zeros((N, N), dtype=float)
    n_samples = 0
    
    best_S = entropy_history[0]
    best_graph = state.get_graph().copy()

    for sweep in range(1, n_sweeps + 1):
        dS, nmoves = state.mcmc_sweep(niter=10)
        S_cur = state.entropy()
        entropy_history.append(S_cur)
        
        if S_cur < best_S:
            best_S = S_cur
            best_graph = state.get_graph().copy()
            
        if sweep > burn_in:
            g = state.get_graph()
            for e in g.edges():
                src, tgt = int(e.source()), int(e.target())
                if src != tgt:
                    edge_counts[tgt, src] += 1.0
            n_samples += 1
            
        if sweep % 10 == 0 or sweep == 1:
            print(f"          [Sweep {sweep:2d}/{n_sweeps}] Entropy = {S_cur:8.2f} (dS = {dS:8.2f})", flush=True)

    P_edge = edge_counts / max(1, n_samples)
    elapsed = time.time() - start_t

    # Ground truth edges (ignore diagonal)
    mask = ~np.eye(N, dtype=bool)
    y_true = (W_true[mask] != 0).astype(int)
    y_score = P_edge[mask]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    pr_auc = float(average_precision_score(y_true, y_score))

    # Metrics from posterior probability threshold P_edge >= 0.5
    W_inf_prob = (P_edge >= 0.5).astype(int)
    np.fill_diagonal(W_inf_prob, 0)
    metrics_prob = compute_reconstruction_metrics(W_true, W_inf_prob)

    # Metrics from best graph (minimum entropy)
    W_inf_best = np.zeros((N, N), dtype=int)
    for e in best_graph.edges():
        src, tgt = int(e.source()), int(e.target())
        if src != tgt:
            W_inf_best[tgt, src] = 1
    metrics_best = compute_reconstruction_metrics(W_true, W_inf_best)

    diag = compute_physical_diagnostics(s, W_true, dt_meas=dt_meas, noise_std=noise_std)

    result = {
        "label": label,
        "N": N,
        "M": M,
        "dt_meas": dt_meas,
        "noise_std": noise_std,
        "n_sweeps": n_sweeps,
        "burn_in": burn_in,
        "elapsed_sec": round(elapsed, 2),
        "best_entropy": float(best_S),
        "AUROC": round(roc_auc, 4),
        "AUPRC": round(pr_auc, 4),
        "prob_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics_prob.items()},
        "best_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics_best.items()},
        "diagnostics": {k: round(v, 6) for k, v in diag.items()}
    }
    return result

def experiment_1_measurement_interval(W_true):
    """
    Experiment 1: Measurement Interval (dt_meas)
    T_total = 100.0, dt = 0.01, dt_meas in [0.05, 0.1, 0.2, 0.5, 1.0]
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Measurement Interval (dt_meas)", flush=True)
    print("=" * 70, flush=True)
    
    dt_meas_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    
    for dt_m in dt_meas_values:
        print(f"\n[Exp 1] Simulating T_total=100.0, dt=0.01, dt_meas={dt_m}...", flush=True)
        s, _ = simulate_grn_sde_trajectory(
            W_true, T_total=100.0, dt=0.01, dt_meas=dt_m,
            d=0.5, V=2.0, noise_std=0.15, seed=424
        )
        print(f"        -> Measured shape: {s.shape} (M={s.shape[1]})", flush=True)
        res = run_single_experiment(
            W_true, s, dt_meas=dt_m, noise_std=0.15,
            n_sweeps=30, burn_in=10, label=f"dt_meas={dt_m}"
        )
        print(f"        -> AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f} | F1 (prob): {res['prob_metrics']['F1-Score']:.4f} | SHD: {res['prob_metrics']['SHD']} | Elapsed: {res['elapsed_sec']}s", flush=True)
        results.append(res)
        
    return results

def experiment_2_trajectory_length(W_true):
    """
    Experiment 2: Trajectory Length / Sample Size (M)
    Fixed dt_meas = 0.1, M in [50, 100, 200, 500, 1000]
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: Trajectory Length / Sample Size (M)", flush=True)
    print("=" * 70, flush=True)
    
    dt_m = 0.1
    M_values = [50, 100, 200, 500, 1000]
    results = []
    
    for M in M_values:
        T_tot = M * dt_m
        print(f"\n[Exp 2] Simulating M={M}, dt_meas={dt_m} (T_total={T_tot:.1f})...", flush=True)
        s, _ = simulate_grn_sde_trajectory(
            W_true, T_total=T_tot, dt=0.01, dt_meas=dt_m,
            d=0.5, V=2.0, noise_std=0.15, seed=424
        )
        print(f"        -> Measured shape: {s.shape} (M={s.shape[1]})", flush=True)
        res = run_single_experiment(
            W_true, s, dt_meas=dt_m, noise_std=0.15,
            n_sweeps=30, burn_in=10, label=f"M={M}"
        )
        print(f"        -> AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f} | F1 (prob): {res['prob_metrics']['F1-Score']:.4f} | SHD: {res['prob_metrics']['SHD']} | Elapsed: {res['elapsed_sec']}s", flush=True)
        results.append(res)
        
    return results

def experiment_3_noise_intensity(W_true):
    """
    Experiment 3: Noise Intensity (noise_std)
    Fixed T_total = 100.0, dt = 0.01, dt_meas = 0.1, noise_std in [0.05, 0.15, 0.30, 0.50]
    """
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: Noise Intensity (noise_std)", flush=True)
    print("=" * 70, flush=True)
    
    noise_values = [0.05, 0.15, 0.30, 0.50]
    results = []
    
    for n_std in noise_values:
        print(f"\n[Exp 3] Simulating T_total=100.0, dt_meas=0.1, noise_std={n_std}...", flush=True)
        s, _ = simulate_grn_sde_trajectory(
            W_true, T_total=100.0, dt=0.01, dt_meas=0.1,
            d=0.5, V=2.0, noise_std=n_std, seed=424
        )
        print(f"        -> Measured shape: {s.shape} (M={s.shape[1]})", flush=True)
        res = run_single_experiment(
            W_true, s, dt_meas=0.1, noise_std=n_std,
            n_sweeps=30, burn_in=10, label=f"noise_std={n_std}"
        )
        diag = res["diagnostics"]
        print(f"        -> AUROC: {res['AUROC']:.4f} | AUPRC: {res['AUPRC']:.4f} | F1 (prob): {res['prob_metrics']['F1-Score']:.4f} | Elapsed: {res['elapsed_sec']}s", flush=True)
        print(f"        -> MeanVar(x): {diag['mean_var_x']:.4f} | Empirical SNR: {diag['empirical_snr']:.4f}", flush=True)
        results.append(res)
        
    return results

def generate_plots(res1, res2, res3, out_dir):
    """
    Generates 3 publication figures in out_dir:
      - fig_exp1_dt_meas.png
      - fig_exp2_sample_size.png
      - fig_exp3_noise_intensity.png
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Figure 1: Measurement Interval ---
    dt_vals = [r["dt_meas"] for r in res1]
    auroc1 = [r["AUROC"] for r in res1]
    auprc1 = [r["AUPRC"] for r in res1]
    f1_prob1 = [r["prob_metrics"]["F1-Score"] for r in res1]
    shd1 = [r["prob_metrics"]["SHD"] for r in res1]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(dt_vals, auroc1, "o-", color="#d95f02", linewidth=2.5, markersize=7, label="AUROC")
    ax1.plot(dt_vals, auprc1, "s-", color="#1b9e77", linewidth=2.5, markersize=7, label="AUPRC")
    ax1.plot(dt_vals, f1_prob1, "^-", color="#2b5c8f", linewidth=2.5, markersize=7, label="F1-Score ($P_{\\text{edge}} \\geq 0.5$)")
    ax1.set_xlabel("Measurement Interval ($dt_{\\text{meas}}$)")
    ax1.set_ylabel("Inference Metric Score")
    ax1.set_title("Exp 1: GRN Inference Accuracy vs. Measurement Interval ($dt_{\\text{meas}}$)")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower left")
    
    ax2 = ax1.twinx()
    ax2.plot(dt_vals, shd1, "d--", color="#e7298a", linewidth=2.0, markersize=6, label="SHD (Right Axis)")
    ax2.set_ylabel("Structural Hamming Distance (SHD)", color="#e7298a")
    ax2.tick_params(axis='y', labelcolor="#e7298a")
    ax2.grid(False)
    
    plt.tight_layout()
    path1 = os.path.join(out_dir, "fig_exp1_dt_meas.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 1: {path1}", flush=True)

    # --- Figure 2: Trajectory Length ---
    M_vals = [r["M"] for r in res2]
    auroc2 = [r["AUROC"] for r in res2]
    auprc2 = [r["AUPRC"] for r in res2]
    f1_prob2 = [r["prob_metrics"]["F1-Score"] for r in res2]
    shd2 = [r["prob_metrics"]["SHD"] for r in res2]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(M_vals, auroc2, "o-", color="#d95f02", linewidth=2.5, markersize=7, label="AUROC")
    ax1.plot(M_vals, auprc2, "s-", color="#1b9e77", linewidth=2.5, markersize=7, label="AUPRC")
    ax1.plot(M_vals, f1_prob2, "^-", color="#2b5c8f", linewidth=2.5, markersize=7, label="F1-Score ($P_{\\text{edge}} \\geq 0.5$)")
    ax1.set_xlabel("Trajectory Sample Size ($M$, with $dt_{\\text{meas}}=0.1$)")
    ax1.set_ylabel("Inference Metric Score")
    ax1.set_title("Exp 2: Sample Complexity for 10-Gene GRN Edge Recovery")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.plot(M_vals, shd2, "d--", color="#e7298a", linewidth=2.0, markersize=6, label="SHD (Right Axis)")
    ax2.set_ylabel("Structural Hamming Distance (SHD)", color="#e7298a")
    ax2.tick_params(axis='y', labelcolor="#e7298a")
    ax2.grid(False)

    plt.tight_layout()
    path2 = os.path.join(out_dir, "fig_exp2_trajectory_length.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 2: {path2}", flush=True)

    # --- Figure 3: Noise Intensity ---
    n_vals = [r["noise_std"] for r in res3]
    auroc3 = [r["AUROC"] for r in res3]
    auprc3 = [r["AUPRC"] for r in res3]
    f1_prob3 = [r["prob_metrics"]["F1-Score"] for r in res3]
    mean_var3 = [r["diagnostics"]["mean_var_x"] for r in res3]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n_vals, auroc3, "o-", color="#d95f02", linewidth=2.5, markersize=7, label="AUROC")
    ax1.plot(n_vals, auprc3, "s-", color="#1b9e77", linewidth=2.5, markersize=7, label="AUPRC")
    ax1.plot(n_vals, f1_prob3, "^-", color="#2b5c8f", linewidth=2.5, markersize=7, label="F1-Score ($P_{\\text{edge}} \\geq 0.5$)")
    ax1.set_xlabel("Stochastic Noise Intensity ($\\text{noise\\_std}$)")
    ax1.set_ylabel("Inference Metric Score")
    ax1.set_title("Exp 3: Physical Tradeoff in Stochastic Noise Intensity")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(n_vals, mean_var3, "v--", color="#7570b3", linewidth=2.0, markersize=6, label="Mean Expression Variance $\\text{Var}(x)$")
    ax2.set_ylabel("Mean Gene Expression Variance $\\text{Var}(x)$", color="#7570b3")
    ax2.tick_params(axis='y', labelcolor="#7570b3")
    ax2.grid(False)

    plt.tight_layout()
    path3 = os.path.join(out_dir, "fig_exp3_noise_intensity.png")
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved Figure 3: {path3}", flush=True)

def print_summary_tables(res1, res2, res3):
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 1: MEASUREMENT INTERVAL (dt_meas)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'dt_meas':<8} | {'M':<6} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'MeanVar(v)':<10} | {'Emp SNR':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in res1:
        d = r["diagnostics"]
        print(f"{r['dt_meas']:<8} | {r['M']:<6} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {r['prob_metrics']['F1-Score']:<9.4f} | {r['prob_metrics']['SHD']:<5} | {d['mean_var_v']:<10.4f} | {d['empirical_snr']:<8.4f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 2: TRAJECTORY LENGTH / SAMPLE SIZE (M) [dt_meas=0.1]", flush=True)
    print("=" * 80, flush=True)
    print(f"{'M':<6} | {'T_total':<8} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'Precision':<9} | {'Recall':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in res2:
        T_tot = r["M"] * r["dt_meas"]
        p_met = r["prob_metrics"]
        print(f"{r['M']:<6} | {T_tot:<8.1f} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {p_met['F1-Score']:<9.4f} | {p_met['SHD']:<5} | {p_met['Precision']:<9.4f} | {p_met['TPR (Recall)']:<8.4f}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY TABLE 3: NOISE INTENSITY (noise_std) [dt_meas=0.1, M=1000]", flush=True)
    print("=" * 80, flush=True)
    print(f"{'noise_std':<10} | {'AUROC':<7} | {'AUPRC':<7} | {'F1 (prob)':<9} | {'SHD':<5} | {'MeanVar(x)':<10} | {'Emp SNR':<8}", flush=True)
    print("-" * 80, flush=True)
    for r in res3:
        d = r["diagnostics"]
        print(f"{r['noise_std']:<10} | {r['AUROC']:<7.4f} | {r['AUPRC']:<7.4f} | {r['prob_metrics']['F1-Score']:<9.4f} | {r['prob_metrics']['SHD']:<5} | {d['mean_var_x']:<10.4f} | {d['empirical_snr']:<8.4f}", flush=True)
    print("=" * 80, flush=True)

def main():
    print("=" * 80, flush=True)
    print("GRN INFERENCE DATA & TRAJECTORY EXPERIMENTER PIPELINE", flush=True)
    print("=" * 80, flush=True)

    out_json = os.path.join(_proj_dir, "results_data_trajectories.json")
    out_dir = os.path.join(_proj_dir, "figures")

    if "--plot-only" in sys.argv and os.path.exists(out_json):
        print(f"Loading existing experimental results from: {out_json}", flush=True)
        with open(out_json, "r") as f:
            all_results = json.load(f)
        res_exp1 = all_results["exp1_dt_meas"]
        res_exp2 = all_results["exp2_sample_size"]
        res_exp3 = all_results["exp3_noise_std"]
        print_summary_tables(res_exp1, res_exp2, res_exp3)
        generate_plots(res_exp1, res_exp2, res_exp3, out_dir)
        print(f"All figures saved to directory: {out_dir}", flush=True)
        print("\nSUCCESS: Plots and tables regenerated cleanly!", flush=True)
        return

    W_true = get_benchmark_10gene_network()
    print(f"Loaded benchmark network: N={W_true.shape[0]}, directed edges={(W_true != 0).sum()}", flush=True)
    all_results = {}

    # Run Experiments with incremental JSON saving
    res_exp1 = experiment_1_measurement_interval(W_true)
    all_results["exp1_dt_meas"] = res_exp1
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    res_exp2 = experiment_2_trajectory_length(W_true)
    all_results["exp2_sample_size"] = res_exp2
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    res_exp3 = experiment_3_noise_intensity(W_true)
    all_results["exp3_noise_std"] = res_exp3
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Summary Tables
    print_summary_tables(res_exp1, res_exp2, res_exp3)
    print(f"\nSaved complete experimental results to: {out_json}", flush=True)

    # Generate Figures
    generate_plots(res_exp1, res_exp2, res_exp3, out_dir)
    print(f"All figures saved to directory: {out_dir}", flush=True)

    print("\nSUCCESS: All data & trajectory experiments completed!", flush=True)

if __name__ == "__main__":
    main()
