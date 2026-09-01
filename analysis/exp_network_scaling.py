#!/usr/bin/env python3
"""
exp_network_scaling.py

Standalone experimental script to investigate how GRN inference accuracy
scales with network size (N), sample complexity (M/N), and network density,
and to identify structural topological sources of error (hub nodes,
reciprocal feedback loops, SBM sparsity prior).

Uses:
  - simulate_grn_sde_trajectory from run_onsager_graph_tool_pipeline.py
  - GeneExprBlockState from gene_expr C++ extension for graph-tool
"""

import os
import sys
import json
import time
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

# Ensure current dir is in sys.path
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from analysis.run_onsager_graph_tool_pipeline import simulate_grn_sde_trajectory


def generate_synthetic_grn(N, avg_degree=1.8, density=None, seed=42):
    """
    Generates a synthetic directed GRN adjacency matrix W (N x N)
    with activating (+1) and repressing (-1) edges, including hub nodes
    and reciprocal feedback loops.
    W[i, j] represents effect of j (source) on i (target).
    """
    rng = np.random.RandomState(seed)
    W = np.zeros((N, N), dtype=np.float32)

    if density is not None:
        target_edges = int(round(density * N * (N - 1)))
    else:
        target_edges = int(round(avg_degree * N))

    target_edges = min(target_edges, N * (N - 1))

    # Create hub-like weights for nodes (exponential distribution)
    node_weights = rng.exponential(scale=1.5, size=N) + 0.5
    node_weights /= node_weights.sum()

    # ~15% of edges as reciprocal loop pairs (i->j and j->i)
    num_reciprocal_pairs = int(round(target_edges * 0.15))
    num_single_edges = target_edges - 2 * num_reciprocal_pairs
    if num_single_edges < 0:
        num_reciprocal_pairs = target_edges // 2

    edges_added = set()

    # 1. Add reciprocal pairs
    attempts = 0
    while len(edges_added) < 2 * num_reciprocal_pairs and attempts < 2000:
        attempts += 1
        i, j = rng.choice(N, size=2, replace=False, p=node_weights)
        if (i, j) not in edges_added and (j, i) not in edges_added:
            edges_added.add((i, j))
            edges_added.add((j, i))

    # 2. Add remaining single directed edges
    attempts = 0
    while len(edges_added) < target_edges and attempts < 5000:
        attempts += 1
        i, j = rng.choice(N, size=2, replace=False, p=node_weights)
        if (i, j) not in edges_added:
            edges_added.add((i, j))

    # Assign weights (~60% activating +1.0, 40% repressing -1.0)
    for i, j in edges_added:
        weight = 1.0 if rng.uniform() < 0.6 else -1.0
        W[i, j] = weight

    return W


def run_inference(W_true, M, n_sweeps=15, burn_in=5, seed=42):
    """
    Simulates SDE trajectory of length M observations and runs annealed MCMC
    inference using GeneExprBlockState. Returns P_edge matrix (N x N) of
    posterior edge probabilities.
    """
    N = W_true.shape[0]
    dt = 0.01
    dt_meas = 0.5
    T_total = M * dt_meas

    s, _ = simulate_grn_sde_trajectory(
        W_true,
        T_total=T_total,
        dt=dt,
        dt_meas=dt_meas,
        d=0.5,
        V=2.0,
        noise_std=0.15,
        seed=seed,
    )

    state = GeneExprBlockState(s, dt=dt_meas)

    edge_counts = np.zeros((N, N), dtype=float)
    n_samples = 0

    for sweep in range(1, n_sweeps + 1):
        alpha = (sweep - 1) / max(1, n_sweeps - 1)
        beta = 0.1 + alpha * 0.9
        state.mcmc_sweep(niter=3, beta=beta)
        if sweep > burn_in:
            g = state.get_graph()
            for e in g.edges():
                src, tgt = int(e.source()), int(e.target())
                if src != tgt:
                    edge_counts[tgt, src] += 1.0
            n_samples += 1

    P_edge = edge_counts / max(1, n_samples)
    return P_edge


def compute_metrics(W_true, P_edge, threshold=0.5):
    """
    Computes structure reconstruction metrics and topological breakdown:
      - AUROC, AUPRC, F1, SHD, Precision, Recall, FPR
      - Hub vs Non-Hub FPR and Recall
      - Reciprocal vs One-Way Recall and posterior probabilities
      - True density vs Inferred density
    """
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)

    y_true = (W_true[mask] != 0).astype(int)
    y_score = P_edge[mask]

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        auprc = float(np.mean(y_true))

    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    shd = int(fp + fn)

    true_density = float(np.sum(y_true) / (N * (N - 1)))
    inferred_density = float(np.sum(y_pred) / (N * (N - 1)))
    expected_density = float(np.sum(y_score) / (N * (N - 1)))

    # --- Topological Breakdown 1: Hub vs Non-Hub ---
    degrees = (W_true != 0).sum(axis=1) + (W_true != 0).sum(axis=0)
    hub_thresh = np.percentile(degrees, 75)
    is_hub = degrees >= max(hub_thresh, 3)

    hub_true_probs, nonhub_true_probs = [], []
    hub_null_probs, nonhub_null_probs = [], []
    hub_tp, hub_fn, hub_fp, hub_tn = 0, 0, 0, 0
    nonhub_tp, nonhub_fn, nonhub_fp, nonhub_tn = 0, 0, 0, 0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            is_edge_hub = is_hub[i] or is_hub[j]
            gt_edge = W_true[i, j] != 0
            p = P_edge[i, j]
            pred_edge = p >= threshold

            if gt_edge:
                if is_edge_hub:
                    hub_true_probs.append(p)
                    if pred_edge:
                        hub_tp += 1
                    else:
                        hub_fn += 1
                else:
                    nonhub_true_probs.append(p)
                    if pred_edge:
                        nonhub_tp += 1
                    else:
                        nonhub_fn += 1
            else:
                if is_edge_hub:
                    hub_null_probs.append(p)
                    if pred_edge:
                        hub_fp += 1
                    else:
                        hub_tn += 1
                else:
                    nonhub_null_probs.append(p)
                    if pred_edge:
                        nonhub_fp += 1
                    else:
                        nonhub_tn += 1

    hub_recall = (
        float(hub_tp / (hub_tp + hub_fn)) if (hub_tp + hub_fn) > 0 else 0.0
    )
    nonhub_recall = (
        float(nonhub_tp / (nonhub_tp + nonhub_fn))
        if (nonhub_tp + nonhub_fn) > 0
        else 0.0
    )
    hub_fpr = (
        float(hub_fp / (hub_fp + hub_tn)) if (hub_fp + hub_tn) > 0 else 0.0
    )
    nonhub_fpr = (
        float(nonhub_fp / (nonhub_fp + nonhub_tn))
        if (nonhub_fp + nonhub_tn) > 0
        else 0.0
    )

    # --- Topological Breakdown 2: Reciprocal vs One-Way vs Reverse Null ---
    reciprocal_true_probs = []
    oneway_true_probs = []
    reverse_null_probs = []
    unrelated_null_probs = []

    reciprocal_tp, reciprocal_fn = 0, 0
    oneway_tp, oneway_fn = 0, 0

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            gt_fwd = W_true[i, j] != 0
            gt_rev = W_true[j, i] != 0
            p = P_edge[i, j]
            pred = p >= threshold

            if gt_fwd and gt_rev:
                reciprocal_true_probs.append(p)
                if pred:
                    reciprocal_tp += 1
                else:
                    reciprocal_fn += 1
            elif gt_fwd and not gt_rev:
                oneway_true_probs.append(p)
                if pred:
                    oneway_tp += 1
                else:
                    oneway_fn += 1
            elif not gt_fwd and gt_rev:
                reverse_null_probs.append(p)
            else:
                unrelated_null_probs.append(p)

    reciprocal_recall = (
        float(reciprocal_tp / (reciprocal_tp + reciprocal_fn))
        if (reciprocal_tp + reciprocal_fn) > 0
        else 0.0
    )
    oneway_recall = (
        float(oneway_tp / (oneway_tp + oneway_fn))
        if (oneway_tp + oneway_fn) > 0
        else 0.0
    )

    mean_p_reciprocal = (
        float(np.mean(reciprocal_true_probs)) if reciprocal_true_probs else 0.0
    )
    mean_p_oneway = (
        float(np.mean(oneway_true_probs)) if oneway_true_probs else 0.0
    )
    mean_p_rev_null = (
        float(np.mean(reverse_null_probs)) if reverse_null_probs else 0.0
    )
    mean_p_unrel_null = (
        float(np.mean(unrelated_null_probs)) if unrelated_null_probs else 0.0
    )

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "F1": f1,
        "SHD": shd,
        "True_Density": true_density,
        "Inferred_Density": inferred_density,
        "Expected_Density": expected_density,
        "Hub_Recall": hub_recall,
        "NonHub_Recall": nonhub_recall,
        "Hub_FPR": hub_fpr,
        "NonHub_FPR": nonhub_fpr,
        "Reciprocal_Recall": reciprocal_recall,
        "OneWay_Recall": oneway_recall,
        "Mean_P_Reciprocal_True": mean_p_reciprocal,
        "Mean_P_OneWay_True": mean_p_oneway,
        "Mean_P_Reverse_Null": mean_p_rev_null,
        "Mean_P_Unrelated_Null": mean_p_unrel_null,
    }


def run_experiment_1():
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: Network Size Scaling (N = 5, 10, 15, 20)", flush=True)
    print("=" * 70, flush=True)

    sizes = [5, 10, 15, 20]
    seeds = [42]
    M = 200

    results = []

    for N in sizes:
        print(
            f"\n--- Testing Network Size N = {N} (M = {M}) ---", flush=True
        )
        metrics_list = []
        for seed in seeds:
            t0 = time.time()
            W_true = generate_synthetic_grn(N, avg_degree=1.8, seed=seed)
            P_edge = run_inference(
                W_true, M=M, n_sweeps=15, burn_in=5, seed=seed * 10
            )
            m = compute_metrics(W_true, P_edge)
            dt = time.time() - t0
            metrics_list.append(m)
            print(
                f"  [Seed {seed}] AUROC={m['AUROC']:.3f} | AUPRC={m['AUPRC']:.3f} | "
                f"F1={m['F1']:.3f} | SHD={m['SHD']} | ({dt:.1f}s)",
                flush=True,
            )

        avg_m = {
            k: float(np.mean([x[k] for x in metrics_list]))
            for k in metrics_list[0]
        }
        std_m = {
            k: float(np.std([x[k] for x in metrics_list]))
            for k in metrics_list[0]
        }
        results.append(
            {"N": N, "M": M, "mean": avg_m, "std": std_m, "runs": metrics_list}
        )

        print(
            f"  -> Mean AUROC: {avg_m['AUROC']:.3f} +/- {std_m['AUROC']:.3f}",
            flush=True,
        )
        print(
            f"  -> Mean AUPRC: {avg_m['AUPRC']:.3f} +/- {std_m['AUPRC']:.3f}",
            flush=True,
        )
        print(
            f"  -> Mean SHD:   {avg_m['SHD']:.1f} +/- {std_m['SHD']:.1f}",
            flush=True,
        )

    return results


def run_experiment_2():
    print("\n" + "=" * 70, flush=True)
    print(
        "EXPERIMENT 2: Sample Complexity Ratio (M / N in [10, 20, 40, 80])",
        flush=True,
    )
    print("=" * 70, flush=True)

    sizes = [10, 15]
    ratios = [10, 20, 40, 80]
    seeds = [101]

    results = []

    for N in sizes:
        print(
            f"\n==================== Network Size N = {N} ====================",
            flush=True,
        )
        for r in ratios:
            M = int(r * N)
            print(
                f"\n--- Testing Ratio M / N = {r} (M = {M} observations) ---",
                flush=True,
            )
            metrics_list = []
            for seed in seeds:
                t0 = time.time()
                W_true = generate_synthetic_grn(N, avg_degree=1.8, seed=seed)
                P_edge = run_inference(
                    W_true, M=M, n_sweeps=15, burn_in=5, seed=seed * 5
                )
                m = compute_metrics(W_true, P_edge)
                dt = time.time() - t0
                metrics_list.append(m)
                print(
                    f"  [Seed {seed}] M/N={r} | AUROC={m['AUROC']:.3f} | AUPRC={m['AUPRC']:.3f} | "
                    f"F1={m['F1']:.3f} | SHD={m['SHD']} | ({dt:.1f}s)",
                    flush=True,
                )

            avg_m = {
                k: float(np.mean([x[k] for x in metrics_list]))
                for k in metrics_list[0]
            }
            std_m = {
                k: float(np.std([x[k] for x in metrics_list]))
                for k in metrics_list[0]
            }
            results.append(
                {
                    "N": N,
                    "ratio": r,
                    "M": M,
                    "mean": avg_m,
                    "std": std_m,
                    "runs": metrics_list,
                }
            )

            print(
                f"  -> Mean AUROC: {avg_m['AUROC']:.3f} +/- {std_m['AUROC']:.3f} "
                f"| F1: {avg_m['F1']:.3f}",
                flush=True,
            )

    return results


def run_experiment_3():
    print("\n" + "=" * 70, flush=True)
    print(
        "EXPERIMENT 3: Topological Failure Modes & Network Density Scaling",
        flush=True,
    )
    print("=" * 70, flush=True)

    print(
        "\n--- [3A] Density Scaling (N = 10, M = 200, D in [0.10, 0.20, 0.35, 0.50]) ---",
        flush=True,
    )
    N = 10
    M = 200
    densities = [0.10, 0.20, 0.35, 0.50]
    seeds = [301]

    density_results = []
    for D in densities:
        print(f"\n--- Testing Density D = {D:.2f} ---", flush=True)
        metrics_list = []
        for seed in seeds:
            t0 = time.time()
            W_true = generate_synthetic_grn(
                N, density=D, seed=seed
            )
            P_edge = run_inference(
                W_true, M=M, n_sweeps=15, burn_in=5, seed=seed * 3
            )
            m = compute_metrics(W_true, P_edge)
            dt = time.time() - t0
            metrics_list.append(m)
            print(
                f"  [Seed {seed}] True D={m['True_Density']:.3f} | Inf D={m['Inferred_Density']:.3f} | "
                f"Exp D={m['Expected_Density']:.3f} | AUROC={m['AUROC']:.3f} | SHD={m['SHD']} ({dt:.1f}s)",
                flush=True,
            )

        avg_m = {
            k: float(np.mean([x[k] for x in metrics_list]))
            for k in metrics_list[0]
        }
        std_m = {
            k: float(np.std([x[k] for x in metrics_list]))
            for k in metrics_list[0]
        }
        density_results.append(
            {
                "density_target": D,
                "mean": avg_m,
                "std": std_m,
                "runs": metrics_list,
            }
        )

        print(
            f"  -> Mean True D: {avg_m['True_Density']:.3f} | Mean Inf D: {avg_m['Inferred_Density']:.3f} | "
            f"FPR={avg_m['FPR']:.3f} | Recall={avg_m['Recall']:.3f}",
            flush=True,
        )

    return {"density_scaling": density_results}


def main():
    start_time = time.time()
    print("=" * 72, flush=True)
    print(
        "STARTING GRN NETWORK SCALING & TOPOLOGY INVESTIGATION", flush=True
    )
    print("=" * 72, flush=True)

    exp1_res = run_experiment_1()
    exp2_res = run_experiment_2()
    exp3_res = run_experiment_3()

    all_results = {
        "exp1_size_scaling": exp1_res,
        "exp2_sample_complexity": exp2_res,
        "exp3_topological_failure_modes": exp3_res,
    }

    out_file = os.path.join(_proj_dir, "scaling_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 72, flush=True)
    print(
        f"SUCCESS: All experiments completed in {total_time:.1f}s! Results saved to {out_file}",
        flush=True,
    )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
