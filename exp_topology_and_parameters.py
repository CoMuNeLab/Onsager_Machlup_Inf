#!/usr/bin/env python3
"""
Topology and Parameter Scaling Experiment
Investigates the impact of:
1. Topology (E-R, Scale-Free, SBM)
2. Feedback Loops (2-node and 3-node)
3. Parameter Scales (Same vs Different)
on Bayesian GRN inference.
"""

import os
import sys
import json
import time
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure current dir is in sys.path
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from run_onsager_graph_tool_pipeline import simulate_grn_sde_trajectory

def generate_base_topology(N, topo_type, density, seed):
    """
    Generates a directed adjacency matrix of size N x N with a target density.
    """
    rng = np.random.RandomState(seed)
    target_edges = int(round(density * N * (N - 1)))
    
    if topo_type == 'ER':
        G = nx.gnp_random_graph(N, density, directed=True, seed=seed)
    elif topo_type == 'SF':
        # nx.scale_free_graph generates a directed multigraph
        G_multi = nx.scale_free_graph(N, seed=seed)
        G = nx.DiGraph(G_multi) # remove parallel edges
        G.remove_edges_from(nx.selfloop_edges(G))
    elif topo_type == 'SBM':
        # 3 blocks of roughly equal size
        sizes = [N//3, N//3, N - 2*(N//3)]
        # Make intra-block probability higher than inter-block
        p_in = density * 2.0
        p_out = density * 0.5
        p_matrix = [[p_in, p_out, p_out],
                    [p_out, p_in, p_out],
                    [p_out, p_out, p_in]]
        G = nx.stochastic_block_model(sizes, p_matrix, directed=True, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topo_type}")
        
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Adjust edges to exactly match target_edges
    edges = list(G.edges())
    while len(edges) > target_edges:
        e = rng.choice(len(edges))
        G.remove_edge(*edges.pop(e))
        
    while len(edges) < target_edges:
        u, v = rng.choice(N, 2, replace=False)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            edges.append((u, v))
            
    return nx.to_numpy_array(G)

def inject_feedback_loops(W, target_edges, fl_level, seed):
    """
    Injects 2-node and 3-node feedback loops.
    fl_level: fraction of target_edges that should be part of a loop.
    Returns signed W.
    """
    rng = np.random.RandomState(seed)
    N = W.shape[0]
    
    num_loop_edges = int(round(target_edges * fl_level))
    num_2node_edges = num_loop_edges // 2
    num_3node_edges = num_loop_edges - num_2node_edges
    
    pairs_needed = num_2node_edges // 2
    triplets_needed = num_3node_edges // 3
    
    W_new = np.zeros_like(W)
    edges_added = set()
    
    # Inject 2-node loops (i <-> j)
    attempts = 0
    while pairs_needed > 0 and attempts < 1000:
        attempts += 1
        i, j = rng.choice(N, 2, replace=False)
        if (i, j) not in edges_added and (j, i) not in edges_added:
            edges_added.add((i, j))
            edges_added.add((j, i))
            pairs_needed -= 1
            
    # Inject 3-node loops (i -> j -> k -> i)
    attempts = 0
    while triplets_needed > 0 and attempts < 1000:
        attempts += 1
        i, j, k = rng.choice(N, 3, replace=False)
        if (i, j) not in edges_added and (j, k) not in edges_added and (k, i) not in edges_added:
            if (j, i) not in edges_added and (k, j) not in edges_added and (i, k) not in edges_added:
                edges_added.add((i, j))
                edges_added.add((j, k))
                edges_added.add((k, i))
                triplets_needed -= 1
                
    # Add remaining single edges from base topology if possible, or random
    base_edges = [(i, j) for i in range(N) for j in range(N) if W[i, j] > 0]
    rng.shuffle(base_edges)
    
    for (i, j) in base_edges:
        if len(edges_added) >= target_edges:
            break
        if (i, j) not in edges_added:
            edges_added.add((i, j))
            
    while len(edges_added) < target_edges:
        i, j = rng.choice(N, 2, replace=False)
        if (i, j) not in edges_added:
            edges_added.add((i, j))
            
    # Assign signs (+1 / -1)
    for (i, j) in edges_added:
        W_new[i, j] = 1.0 if rng.uniform() < 0.6 else -1.0
        
    return W_new

def sample_parameters(N, scale_mode, seed):
    """
    Samples dynamics parameters d, V, noise_std.
    """
    rng = np.random.RandomState(seed)
    if scale_mode == 'same':
        d = np.clip(rng.normal(0.5, 0.05, size=N), 0.1, 1.0)
        V = np.clip(rng.normal(2.0, 0.2, size=N), 0.5, 4.0)
        noise_std = np.clip(rng.normal(0.15, 0.02, size=N), 0.01, 0.5)
    elif scale_mode == 'different':
        d = rng.lognormal(mean=np.log(0.5), sigma=0.8, size=N)
        V = rng.lognormal(mean=np.log(2.0), sigma=0.8, size=N)
        noise_std = rng.lognormal(mean=np.log(0.15), sigma=0.8, size=N)
    else:
        raise ValueError("Unknown scale_mode")
    return d, V, noise_std

def run_inference(W_true, M, d, V, noise_std, seed=42):
    N = W_true.shape[0]
    dt = 0.01
    dt_meas = 0.5
    T_total = M * dt_meas

    s, _ = simulate_grn_sde_trajectory(
        W_true,
        T_total=T_total,
        dt=dt,
        dt_meas=dt_meas,
        d=d,
        V=V,
        noise_std=noise_std,
        seed=seed,
    )

    state = GeneExprBlockState(s, dt=dt_meas)
    edge_counts = np.zeros((N, N), dtype=float)
    n_samples = 0
    n_sweeps = 15
    burn_in = 5

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
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    shd = int(fp + fn)

    return {
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "SHD": shd,
    }

def main():
    print("=" * 80)
    print("TOPOLOGY AND PARAMETER SCALING EXPERIMENT")
    print("=" * 80)

    # Note: To test the script fast, I am using smaller M and sweeps
    sizes = [10, 50, 100, 200]
    topologies = ['ER', 'SF', 'SBM']
    fl_levels = [0.0, 0.15, 0.30] # 0%, 15%, 30% of edges in feedback loops
    scales = ['same', 'different']
    M = 400
    density = 0.15 # Baseline sparsity

    results = []
    out_file = os.path.join(_proj_dir, "results_topology_experiment.json")

    total_runs = len(sizes) * len(topologies) * len(fl_levels) * len(scales)
    run_idx = 0

    for N in sizes:
        # Adjust density for larger networks to keep average degree reasonable (~3-5)
        # avg_degree = density * (N - 1). If we want avg_degree = 3, density = 3/(N-1)
        current_density = max(3.0 / (N - 1), 0.02)
        target_edges = int(round(current_density * N * (N - 1)))
        
        for topo in topologies:
            for fl in fl_levels:
                for scale in scales:
                    run_idx += 1
                    seed = run_idx * 42
                    
                    # 1. Topology
                    W_base = generate_base_topology(N, topo, current_density, seed)
                    W_true = inject_feedback_loops(W_base, target_edges, fl, seed)
                    
                    # 2. Parameters
                    d, V, noise_std = sample_parameters(N, scale, seed)
                    
                    # 3. Inference
                    t0 = time.time()
                    P_edge = run_inference(W_true, M, d, V, noise_std, seed)
                    metrics = compute_metrics(W_true, P_edge)
                    dt = time.time() - t0
                    
                    res_dict = {
                        "N": N,
                        "Topology": topo,
                        "FL_Level": fl,
                        "Scale": scale,
                        "Metrics": metrics,
                        "Time": dt
                    }
                    results.append(res_dict)
                    
                    print(f"[{run_idx:3d}/{total_runs}] N={N:<3} | Topo={topo:<3} | FL={fl:<4.2f} | Scale={scale:<9} "
                          f"-> AUROC={metrics['AUROC']:.3f} | F1={metrics['F1']:.3f} | SHD={metrics['SHD']} ({dt:.1f}s)")
                    
                    # Save intermediate
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"EXPERIMENT COMPLETE. Results saved to {out_file}")

if __name__ == "__main__":
    main()
