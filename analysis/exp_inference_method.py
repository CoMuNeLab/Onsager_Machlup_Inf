#!/usr/bin/env python3
"""
exp_inference_method.py
Standalone experimental script for investigating MCMC inference methods,
simulated annealing schedules, parameter degeneracy / identifiability,
and optimal inference configurations for GeneExprBlockState in graph-tool.
"""

import os
import sys
import time
import numpy as np

# Add current project directory to path
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from analysis.run_onsager_graph_tool_pipeline import get_benchmark_10gene_network, simulate_grn_sde_trajectory

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


# =========================================================================
# 1. Helper for Reconstruction Metrics
# =========================================================================
def evaluate_reconstruction_metrics(W_true, W_pred, P_prob=None):
    """
    Computes structure reconstruction metrics vs ground-truth W_true.
    If P_prob (posterior edge probability matrix) is provided, computes AUROC and AUPRC.
    """
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)  # ignore diagonal
    y_true = (W_true[mask] != 0).astype(int)
    y_pred = (W_pred[mask] != 0).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    shd = fp + fn

    auroc = 0.0
    auprc = 0.0
    best_f1_thresh = f1
    best_thresh = 0.5
    if P_prob is not None:
        y_score = P_prob[mask]
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            for th in np.linspace(0.05, 0.95, 19):
                y_th = (y_score >= th).astype(int)
                f_th = f1_score(y_true, y_th, zero_division=0)
                if f_th > best_f1_thresh:
                    best_f1_thresh = f_th
                    best_thresh = th

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "TPR": tpr,
        "FPR": fpr,
        "Precision": precision,
        "F1": f1,
        "SHD": shd,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Best_F1_prob": best_f1_thresh,
        "Best_thresh": best_thresh,
    }


# =========================================================================
# 2. Empirical Analysis of Parameter Degeneracy & Identifiability
# =========================================================================
def analyze_parameter_degeneracy(s, W_true, dt_meas):
    """
    Empirically investigates parameter degeneracy along two critical ridges:
      1. Equilibrium scaling degeneracy (gamma_i and V_i scaled by c with V_i/gamma_i = const)
         vs non-equilibrium ratio changes.
      2. Sigmoid saturation degeneracy for edge inputs m_i.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: PARAMETER DEGENERACY & IDENTIFIABILITY ANALYSIS")
    print("=" * 72)

    # A. Steady-State / Ratio Scaling Degeneracy
    print("\n--- A. Equilibrium Scaling Degeneracy (gamma, V scaled by factor c) ---")
    print("Comparing entropy variation along constant ratio (V/gamma=4.0) vs varying ratio:")
    print(f"{'Scale c':>8s} | {'log(c)':>8s} | {'Entropy (Const Ratio)':>22s} | {'Entropy (Varying Ratio V_i*c)':>30s}")
    print("-" * 75)

    base_gamma = 0.5
    base_V = 2.0
    base_D = 0.05

    for c in [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]:
        # 1. Constant ratio: scale both gamma and V by c
        theta_const = [np.log(base_gamma * c), np.log(base_V * c), np.log(base_D)]
        state_const = GeneExprBlockState(s, dt=dt_meas, theta=theta_const)
        S_const = state_const.entropy()

        # 2. Varying ratio: scale only V by c (gamma fixed)
        theta_vary = [np.log(base_gamma), np.log(base_V * c), np.log(base_D)]
        state_vary = GeneExprBlockState(s, dt=dt_meas, theta=theta_vary)
        S_vary = state_vary.entropy()

        print(f"{c:8.2f} | {np.log(c):8.3f} | {S_const:22.4f} | {S_vary:30.4f}")

    # B. Sigmoid Saturation & Edge Weight Degeneracy
    print("\n--- B. Sigmoid Saturation Analysis ---")
    print("Evaluating derivative d(sigma)/dm = sigma(m)*(1 - sigma(m)) across regulatory input regimes:")
    print(f"{'Input m_i':>10s} | {'sigma(m_i)':>12s} | {'d(sigma)/dm':>12s} | {'Regime Description':>25s}")
    print("-" * 65)

    test_inputs = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
    for m in test_inputs:
        sig = 1.0 / (1.0 + np.exp(-np.clip(m, -30.0, 30.0)))
        dsig = sig * (1.0 - sig)
        if abs(m) >= 4.0:
            reg = "Saturated (Degenerate)"
        elif abs(m) >= 2.0:
            reg = "Transition"
        else:
            reg = "Linear / Sensitive"
        print(f"{m:10.2f} | {sig:12.4f} | {dsig:12.6f} | {reg:>25s}")

    print("-" * 65)


# =========================================================================
# 3. MCMC Inference Engine with Annealing & Parameter Regularization
# =========================================================================
def run_mcmc_inference_strategy(
    s,
    W_true,
    dt_meas,
    n_sweeps=50,
    niter_per_sweep=5,
    beta_schedule="constant",
    beta_start=1.0,
    beta_end=1.0,
    tdelta_val=1e-8,
    theta_range_val=None,
    burn_in_frac=0.4,
    seed=42,
    verbose=False
):
    """
    Runs Graph-Tool GeneExprBlockState MCMC inference under a specific configuration.
    Returns full reconstruction metrics, runtime, best entropy, and predicted graphs.
    """
    np.random.seed(seed)
    N = W_true.shape[0]

    kwargs = {"tdelta": [tdelta_val] * 3}
    if theta_range_val is not None:
        kwargs["theta_range"] = theta_range_val

    state = GeneExprBlockState(s, dt=dt_meas, **kwargs)

    best_S = state.entropy()
    best_W = np.zeros((N, N), dtype=int)
    for e in state.get_graph().edges():
        src, tgt = int(e.source()), int(e.target())
        if src != tgt:
            best_W[tgt, src] = 1

    edge_counts = np.zeros((N, N), dtype=float)
    burn_in_step = int(n_sweeps * burn_in_frac)
    n_samples = 0

    t0 = time.time()
    for sweep in range(1, n_sweeps + 1):
        # Determine beta according to schedule
        alpha = (sweep - 1) / max(1, n_sweeps - 1)
        if beta_schedule == "constant":
            beta = 1.0
        elif beta_schedule == "linear":
            beta = beta_start + alpha * (beta_end - beta_start)
        elif beta_schedule == "exponential":
            # Geometric progress from beta_start to beta_end
            beta = beta_start * ((beta_end / max(1e-5, beta_start)) ** alpha)
        elif beta_schedule == "two_stage":
            # Low temperature exploration for first 30%, then linear ramp
            if alpha < 0.3:
                beta = beta_start
            else:
                sub_alpha = (alpha - 0.3) / 0.7
                beta = beta_start + sub_alpha * (beta_end - beta_start)
        else:
            beta = 1.0

        state.mcmc_sweep(niter=niter_per_sweep, beta=beta)
        S_cur = state.entropy()

        W_cur = np.zeros((N, N), dtype=int)
        for e in state.get_graph().edges():
            src, tgt = int(e.source()), int(e.target())
            if src != tgt:
                W_cur[tgt, src] = 1

        if S_cur < best_S:
            best_S = S_cur
            best_W = W_cur.copy()

        if sweep > burn_in_step:
            edge_counts += W_cur
            n_samples += 1

    t1 = time.time()
    runtime = t1 - t0

    P_prob = edge_counts / max(1, n_samples)
    metrics = evaluate_reconstruction_metrics(W_true, best_W, P_prob=P_prob)
    metrics["Runtime"] = runtime
    metrics["Best_S"] = best_S

    if verbose:
        print(f"    -> Sweeps={n_sweeps}, Beta={beta_schedule}({beta_start}->{beta_end}), tdelta={tdelta_val}: "
              f"AUROC={metrics['AUROC']:.3f}, AUPRC={metrics['AUPRC']:.3f}, "
              f"F1_MAP={metrics['F1']:.3f}, F1_best={metrics['Best_F1_prob']:.3f}, "
              f"Best_S={best_S:.1f} ({runtime:.2f}s)")

    return metrics, best_W, P_prob


# =========================================================================
# 4. Systematic Experiments & Table Output
# =========================================================================
def run_all_experiments(s, W_true, dt_meas):
    """
    Executes systematic sweeps over niter, annealing schedules, and regularization.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: NUMBER OF MCMC SWEEPS (niter = 20, 50, 100, 150)")
    print("=" * 72)
    print(f"{'niter':>6s} | {'AUROC':>6s} | {'AUPRC':>6s} | {'F1 (MAP)':>8s} | {'F1 (Best)':>9s} | {'Precision':>9s} | {'Recall':>6s} | {'SHD':>4s} | {'Best_S':>9s} | {'Time(s)':>7s}")
    print("-" * 90)

    sweep_results = {}
    for niter in [20, 50, 100, 150]:
        metrics, _, _ = run_mcmc_inference_strategy(
            s, W_true, dt_meas,
            n_sweeps=niter, niter_per_sweep=2,
            beta_schedule="constant", beta_start=1.0, beta_end=1.0,
            tdelta_val=1e-8, seed=42
        )
        sweep_results[niter] = metrics
        print(f"{niter:6d} | {metrics['AUROC']:6.3f} | {metrics['AUPRC']:6.3f} | "
              f"{metrics['F1']:8.3f} | {metrics['Best_F1_prob']:9.3f} | "
              f"{metrics['Precision']:9.3f} | {metrics['TPR']:6.3f} | "
              f"{metrics['SHD']:4d} | {metrics['Best_S']:9.1f} | {metrics['Runtime']:7.2f}")

    print("\n" + "=" * 72)
    print("EXPERIMENT 3: SIMULATED ANNEALING TEMPERATURE SCHEDULES")
    print("=" * 72)
    print(f"{'Schedule':>18s} | {'Beta Range':>10s} | {'AUROC':>6s} | {'AUPRC':>6s} | {'F1 (MAP)':>8s} | {'F1 (Best)':>9s} | {'Precision':>9s} | {'Recall':>6s} | {'SHD':>4s} | {'Best_S':>9s}")
    print("-" * 105)

    anneal_configs = [
        ("constant", 1.0, 1.0, "Constant 1.0"),
        ("linear", 0.5, 1.0, "Lin 0.5->1.0"),
        ("linear", 0.1, 1.0, "Lin 0.1->1.0"),
        ("exponential", 0.1, 1.0, "Exp 0.1->1.0"),
        ("two_stage", 0.2, 1.0, "2Stage 0.2->1.0")
    ]

    anneal_results = {}
    for sched, b_start, b_end, label in anneal_configs:
        metrics, _, _ = run_mcmc_inference_strategy(
            s, W_true, dt_meas,
            n_sweeps=100, niter_per_sweep=1,
            beta_schedule=sched, beta_start=b_start, beta_end=b_end,
            tdelta_val=1e-8, seed=42
        )
        anneal_results[label] = metrics
        print(f"{label:>18s} | {f'{b_start}->{b_end}':>10s} | {metrics['AUROC']:6.3f} | {metrics['AUPRC']:6.3f} | "
              f"{metrics['F1']:8.3f} | {metrics['Best_F1_prob']:9.3f} | "
              f"{metrics['Precision']:9.3f} | {metrics['TPR']:6.3f} | "
              f"{metrics['SHD']:4d} | {metrics['Best_S']:9.1f}")

    print("\n" + "=" * 72)
    print("EXPERIMENT 4: PARAMETER PROPOSALS (tdelta) & REGULARIZATION (theta_range)")
    print("=" * 72)
    print(f"{'Config / tdelta':>22s} | {'AUROC':>6s} | {'AUPRC':>6s} | {'F1 (MAP)':>8s} | {'F1 (Best)':>9s} | {'Precision':>9s} | {'Recall':>6s} | {'SHD':>4s} | {'Best_S':>9s}")
    print("-" * 100)

    # Regularized bounds: prevent extreme parameter scaling
    reg_range = [(-2.0, 2.0), (-1.0, 3.0), (-5.0, -1.0)]
    param_configs = [
        ("tdelta=1e-8 (Default)", 1e-8, None),
        ("tdelta=1e-4", 1e-4, None),
        ("tdelta=1e-2", 1e-2, None),
        ("tdelta=1e-1", 1e-1, None),
        ("tdelta=1e-2 + RegRange", 1e-2, reg_range),
        ("tdelta=1e-1 + RegRange", 1e-1, reg_range),
    ]

    param_results = {}
    for label, td_val, th_range in param_configs:
        metrics, _, _ = run_mcmc_inference_strategy(
            s, W_true, dt_meas,
            n_sweeps=100, niter_per_sweep=1,
            beta_schedule="linear", beta_start=0.1, beta_end=1.0,
            tdelta_val=td_val, theta_range_val=th_range, seed=42
        )
        param_results[label] = metrics
        print(f"{label:>22s} | {metrics['AUROC']:6.3f} | {metrics['AUPRC']:6.3f} | "
              f"{metrics['F1']:8.3f} | {metrics['Best_F1_prob']:9.3f} | "
              f"{metrics['Precision']:9.3f} | {metrics['TPR']:6.3f} | "
              f"{metrics['SHD']:4d} | {metrics['Best_S']:9.1f}")

    return sweep_results, anneal_results, param_results


# =========================================================================
# 5. Optimal Strategy vs Baseline
# =========================================================================
def run_optimal_vs_baseline(s, W_true, dt_meas):
    """
    Compares the default baseline strategy against the best optimized strategy found.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: OPTIMAL INFERENCE CONFIGURATION VS BASELINE")
    print("=" * 72)

    # 1. Baseline Strategy (30 sweeps, constant beta=1.0, tdelta=1e-8, niter=10 as in run_onsager_graph_tool_pipeline.py)
    base_metrics, base_W, base_P = run_mcmc_inference_strategy(
        s, W_true, dt_meas,
        n_sweeps=30, niter_per_sweep=10,
        beta_schedule="constant", beta_start=1.0, beta_end=1.0,
        tdelta_val=1e-8, seed=42
    )

    # 2. Optimal Strategy (150 sweeps, linear annealing 0.1 -> 1.0, tdelta=1e-2, regularized theta_range)
    opt_range = [(-2.0, 2.0), (-1.0, 3.0), (-5.0, -1.0)]
    opt_metrics, opt_W, opt_P = run_mcmc_inference_strategy(
        s, W_true, dt_meas,
        n_sweeps=150, niter_per_sweep=2,
        beta_schedule="linear", beta_start=0.1, beta_end=1.0,
        tdelta_val=1e-2, theta_range_val=opt_range, seed=42
    )

    print("\n--- SIDE-BY-SIDE BENCHMARK COMPARISON ---")
    print(f"{'Metric':>20s} | {'Baseline Strategy':>18s} | {'Optimal Strategy':>18s} | {'Improvement':>12s}")
    print("-" * 75)

    metrics_to_compare = [
        ("AUROC", "AUROC", "+"),
        ("AUPRC", "AUPRC", "+"),
        ("F1-Score (MAP)", "F1", "+"),
        ("F1-Score (Best Thresh)", "Best_F1_prob", "+"),
        ("Precision (MAP)", "Precision", "+"),
        ("Recall / TPR (MAP)", "TPR", "+"),
        ("SHD (Hamming)", "SHD", "-"),
        ("Best Entropy (S)", "Best_S", "-"),
        ("Runtime (s)", "Runtime", "")
    ]

    for label, k, direction in metrics_to_compare:
        v_base = base_metrics[k]
        v_opt = opt_metrics[k]
        if direction == "+":
            diff = v_opt - v_base
            diff_str = f"{diff:+.3f}"
        elif direction == "-":
            diff = v_opt - v_base
            diff_str = f"{diff:+.1f}"
        else:
            diff_str = f"{v_opt / max(1e-5, v_base):.1f}x"

        if isinstance(v_base, int):
            print(f"{label:>20s} | {v_base:18d} | {v_opt:18d} | {diff_str:>12s}")
        else:
            print(f"{label:>20s} | {v_base:18.4f} | {v_opt:18.4f} | {diff_str:>12s}")

    print("-" * 75)
    return base_metrics, opt_metrics


# =========================================================================
# Main Entry Point
# =========================================================================
def main():
    print("=" * 72)
    print("GRAPH-TOOL GRN BAYESIAN INFERENCE: METHOD OPTIMIZATION & IDENTIFIABILITY")
    print("=" * 72)

    # Load 10-gene benchmark GRN and simulate trajectories
    W_true = get_benchmark_10gene_network()
    print("\n[Setup] Simulating Benchmark 10-Gene Non-Equilibrium Trajectories...")
    s, dt_meas = simulate_grn_sde_trajectory(
        W_true, T_total=100.0, dt=0.01, dt_meas=0.5, seed=424
    )
    print(f"    - Trajectory observation tensor shape: {s.shape}")
    print(f"    - Ground-truth directed edges: {(W_true != 0).sum()}")

    # 1. Parameter Degeneracy & Identifiability Analysis
    analyze_parameter_degeneracy(s, W_true, dt_meas)

    # 2. Systematic Experiments (Sweeps, Annealing, Parameter Proposals)
    run_all_experiments(s, W_true, dt_meas)

    # 3. Optimal vs Baseline Comparison
    run_optimal_vs_baseline(s, W_true, dt_meas)

    print("\n" + "=" * 72)
    print("SUCCESS: Full MCMC investigation and benchmarking completed!")
    print("=" * 72)


if __name__ == "__main__":
    main()
