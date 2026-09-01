#!/usr/bin/env python3
"""
Experiment: Information Content of GRN Trajectories.

Computes node-wise and network-averaged information content metrics:
- TDMI: Time-Delayed Mutual Information
- eSNR: Empirical Signal-to-Noise Ratio (using Savitzky-Golay smoothing)
- TIC: Trajectory Information Capacity = log(1+Var) * TDMI

Explores fine-grained ranges of parameters (N, d, V, noise, dt_meas)
and structural properties (topology, degree).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import savgol_filter
import graph_tool.all as gt
from itertools import product
from tqdm import tqdm

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def generate_network(N, topo_type):
    if topo_type == "ER":
        # Erdos-Renyi like directed graph with probability p=2/N
        p = min(1.0, 2.0 / N)
        W = np.random.choice([0, 1, -1], size=(N, N), p=[1-p, p/2, p/2]).astype(np.float32)
        np.fill_diagonal(W, 0)
    elif topo_type == "BA":
        # Preferential attachment (scale free)
        g = gt.price_network(N, m=2, directed=True)
        W = np.zeros((N, N), dtype=np.float32)
        for e in g.edges():
            s, t = int(e.source()), int(e.target())
            W[t, s] = np.random.choice([-1, 1])
    elif topo_type == "SBM":
        # simple manual block structure
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    if (i < N//2 and j < N//2) or (i >= N//2 and j >= N//2):
                        p = 4.0 / N
                    else:
                        p = 0.5 / N
                    if np.random.rand() < p:
                        W[i, j] = np.random.choice([-1, 1])
    else:
        raise ValueError("Unknown topology")
    return W

def simulate_trajectory(W, T_total=50.0, dt_sim=0.01, dt_meas=0.1, d=0.5, V=2.0, noise_std=0.15):
    N = W.shape[0]
    n_steps = int(T_total / dt_sim)
    meas_step = max(1, int(dt_meas / dt_sim))
    
    x = np.random.uniform(0.1, 0.5, size=N).astype(np.float32)
    sqrt_dt_noise = noise_std * np.sqrt(dt_sim)
    
    trajectory = []
    
    for k in range(n_steps):
        if k % meas_step == 0:
            trajectory.append(x.copy())
            
        x_clipped = np.maximum(x, 0.0)
        prod = sigmoid(W @ x_clipped)
        drift = -d * x_clipped + V * prod
        noise = np.random.normal(0.0, 1.0, size=N)
        x = np.maximum(x + dt_sim * drift + sqrt_dt_noise * noise, 0.01)
        
    return np.array(trajectory)

def compute_node_metrics(x_i):
    """
    Computes information content metrics for a single 1D trajectory x_i.
    """
    n_samples = len(x_i)
    if n_samples < 10:
        return 0.0, 0.0, 0.0, 0.0

    var_x = np.var(x_i)
    if var_x < 1e-10:
        return 0.0, 0.0, 0.0, var_x
        
    # TDMI
    tdmi = mutual_info_regression(x_i[:-1].reshape(-1, 1), x_i[1:], random_state=42)[0]
    
    # eSNR
    window_length = min(15, n_samples // 2 * 2 + 1)
    if window_length >= 5:
        try:
            x_smooth = savgol_filter(x_i, window_length=window_length, polyorder=2)
            noise = x_i - x_smooth
            var_smooth = np.var(x_smooth)
            var_noise = np.var(noise)
            esnr = var_smooth / (var_noise + 1e-10)
        except Exception:
            esnr = 0.0
    else:
        esnr = 0.0
        
    # TIC
    tic = np.log1p(var_x) * tdmi
    
    return tdmi, esnr, tic, var_x

def run_experiment():
    Ns = [10, 20, 50]
    topos = ["ER", "BA", "SBM"]
    ds = [0.2, 0.5, 1.0]
    Vs = [1.0, 2.0, 5.0]
    noises = [0.05, 0.15, 0.3]
    dt_meases = [0.05, 0.1, 0.5, 1.0, 2.0]
    
    results = []
    
    param_grid = list(product(Ns, topos, ds, Vs, noises, dt_meases))
    print(f"Total parameter combinations: {len(param_grid)}")
    
    for N, topo, d, V, noise, dt_meas in tqdm(param_grid):
        W = generate_network(N, topo)
        
        in_degrees = np.sum(W != 0, axis=1)
        out_degrees = np.sum(W != 0, axis=0)
        
        # Adaptive time window based on decay rate
        tau = 1.0 / d
        T_transient_end = 5.0 * tau
        T_total = T_transient_end + 35.0
        
        traj = simulate_trajectory(W, T_total=T_total, dt_sim=0.005, dt_meas=dt_meas, 
                                   d=d, V=V, noise_std=noise)
        
        # Split into transient and stationary
        idx_trans = int(T_transient_end / dt_meas)
        traj_trans = traj[:idx_trans, :]
        traj_stat = traj[idx_trans:, :]
        
        for regime, data in [("Transient", traj_trans), ("Stationary", traj_stat)]:
            if len(data) < 10:
                continue
                
            for i in range(N):
                tdmi, esnr, tic, var_x = compute_node_metrics(data[:, i])
                
                results.append({
                    "N": N,
                    "Topology": topo,
                    "d": d,
                    "V": V,
                    "Noise": noise,
                    "dt_meas": dt_meas,
                    "Regime": regime,
                    "Node_ID": i,
                    "In_Degree": in_degrees[i],
                    "Out_Degree": out_degrees[i],
                    "TDMI": tdmi,
                    "eSNR": esnr,
                    "TIC": tic,
                    "Var": var_x
                })
                
    df = pd.DataFrame(results)
    df.to_csv("results_information_content.csv", index=False)
    print("Experiment finished. Results saved to results_information_content.csv")

if __name__ == "__main__":
    run_experiment()
