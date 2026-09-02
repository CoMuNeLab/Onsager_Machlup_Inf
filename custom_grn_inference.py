#!/usr/bin/env python
# coding: utf-8

# # Custom GRN Topology Generation and Graph-Tool Inference
# 
# This notebook allows you to generate custom Gene Regulatory Networks (GRNs) using a directed Scale-Free topology, simulate their trajectories with heterogeneous kinetics, and perform Bayesian inference using the `gene_expr` extension for `graph-tool`.

# In[55]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path to load gene_expr
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))

import graph_tool.all as gt
from gene_expr import GeneExprBlockState


# In[56]:


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


# In[57]:


def generate_scale_free_W(N, m=2, seed=42):
    """Generates a scale-free directed GRN using directed preferential attachment."""
    np.random.seed(seed)
    W = np.zeros((N, N))
    out_degrees = np.zeros(N)
    
    core_size = m + 1
    for i in range(core_size):
        for j in range(core_size):
            if i != j and np.random.rand() > 0.5:
                sign = np.random.choice([-1, 1])
                mag = np.random.uniform(1.0, 3.0)
                W[i, j] = sign * mag
                out_degrees[j] += 1
                
    for i in range(core_size, N):
        weights = out_degrees[:i] + 1.0
        probs = weights / np.sum(weights)
        regulators = np.random.choice(i, size=m, replace=False, p=probs)
        
        for reg in regulators:
            sign = np.random.choice([-1, 1])
            mag = np.random.uniform(1.0, 3.0)
            W[i, reg] = sign * mag
            out_degrees[reg] += 1
            
    return W


# In[58]:


def simulate_realistic_pairs(W, M=200, dt_sim=0.01, dt_sample=0.5, dt_meas=0.05, D_base=0.1, mean_d=0.5, std_d=0.5, mean_V=2.0, std_V=0.5, seed=123, t_window=(0.0, 3.0)):
    N = W.shape[0]
    np.random.seed(seed)
    
    # Heterogeneous parameters
    d = np.random.lognormal(mean=np.log(mean_d), sigma=std_d, size=N)
    V = np.random.lognormal(mean=np.log(mean_V), sigma=std_V, size=N)
    
    t_start, t_end = t_window
    sample_times = np.arange(t_start, t_end + 1e-9, dt_sample)
    if len(sample_times) == 0:
        sample_times = np.array([t_start])
    K = len(sample_times)
    
    n_trajectories = int(np.ceil(M / K))
    
    T_max = t_end + dt_meas
    n_steps = int(np.round(T_max / dt_sim))
    
    sample_step_indices = [int(np.round(t / dt_sim)) for t in sample_times]
    next_step_indices = [int(np.round((t + dt_meas) / dt_sim)) for t in sample_times]
    
    all_pairs = []
    
    for traj_idx in range(n_trajectories):
        x = np.random.uniform(0.1, 1.0, size=N).astype(np.float64)
        history = {0: x.copy()}
        
        for step in range(1, n_steps + 1):
            drift = -d * x + V * sigmoid(np.dot(W, x))
            noise = np.sqrt(D_base * np.maximum(x, 0.01)) * np.random.normal(0, np.sqrt(dt_sim), size=N)
            x = np.maximum(x + dt_sim * drift + noise, 0.01)
            
            if step in sample_step_indices or step in next_step_indices:
                history[step] = x.copy()
                
        for s_idx, n_idx in zip(sample_step_indices, next_step_indices):
            pair = np.column_stack([history[s_idx], history[n_idx]])
            all_pairs.append(pair)
            if len(all_pairs) == M:
                break
        if len(all_pairs) == M:
            break
            
    out_degrees = (W != 0).sum(axis=0)
    hub_node = np.argmax(out_degrees)
            
    return all_pairs, hub_node, out_degrees, d, V


# ### Custom Parameters & Trajectory Simulation
# Set the parameters and generate the network and trajectories.

# In[71]:


# Network Parameters
N = 10              # Number of genes
m = 2               # Edges to attach from a new node to existing nodes
seed_network = 42   # Random seed for network generation

# Simulation Parameters
M_pairs = 200      # Number of observation pairs to harvest
dt_sim = 0.01      # Simulation integration step
dt_meas = 0.05     # Measurement interval for the pair
dt_sample = 0.5    # Interval between harvested pairs in the same trajectory
t_window = (0.0, 3.0) # ONLY TRANSIENT REGION
D_base = 0.1        # Base multiplicative noise level

# Kinetic Parameters (Mean and Std of Log-Normal distribution)
mean_decay = 0.5    # Average decay rate
std_decay = 0.5     # Heterogeneity in decay rate
mean_V = 2.0        # Average max transcription rate
std_V = 0.5         # Heterogeneity in transcription rate
seed_sim = 123      # Random seed for simulation

# Generate Network
W_true = generate_scale_free_W(N, m, seed=seed_network)

# Simulate Trajectories
pairs, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
    W_true, M=M_pairs, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
    D_base=D_base, mean_d=mean_decay, std_d=std_decay, mean_V=mean_V, std_V=std_V, 
    seed=seed_sim, t_window=t_window
)

# Format for graph-tool inference (list of N x 2 arrays)
s = pairs
print(f"Master Regulator is Node {hub_node} with {out_degrees[hub_node]} outgoing edges.")
print(f"Harvested {len(s)} state-transition pairs from transient windows {t_window}.")


# In[72]:


# Plotting disabled because we are no longer generating continuous trajectories, but discrete pairs.


# ### Graph-Tool Bayesian Inference
# We now use the `GeneExprBlockState` from the graph-tool extension to infer the network topology from the time-series data.
# First, we evaluate the description length (entropy) of the true data-generating configuration to establish a baseline.

# In[73]:


# Construct the true graph structure and weights in graph-tool format
g_true = gt.Graph(directed=True)
g_true.add_vertex(N)
ep_x = g_true.new_edge_property("double")

for i in range(N):
    for j in range(N):
        if W_true[i, j] != 0:
            # j is the source regulator, i is the target regulated gene
            e = g_true.add_edge(j, i)
            ep_x[e] = W_true[i, j]
g_true.ep["x"] = ep_x

# Average True node parameters (the inference extension infers global effective parameters)
theta_true = [np.log(np.mean(d_true)), np.log(np.mean(V_true)), np.log(D_base)]

# FIX: Explicitly pass x=g_true.ep["x"] so the edge weights are correctly registered!
state_true = GeneExprBlockState(s, dt=dt_meas, g=g_true, theta=theta_true, x=g_true.ep["x"])
S_true = state_true.entropy()
print(f"Entropy of the TRUE network and parameters: {S_true:.4f}")


# In[74]:


print("\nInstantiating GeneExprBlockState from scratch (empty network)...")
g_init = g_true.copy()
state = GeneExprBlockState(s, dt=dt_meas, g=g_init, theta=theta_true, x=g_init.ep["x"], state_args=dict(deg_corr=True))
S_init = state.entropy()
print(f"Initial description length (empty network): {S_init:.4f}")

print("\nRunning MCMC Sweeps (Network Structure + Parameters)...")
n_sweeps = 30
best_S = S_init
best_graph = None

for sweep in range(1, n_sweeps + 1):
    dS, nmoves = state.mcmc_sweep(niter=10)
    S_cur = state.entropy()
    if S_cur < best_S or best_graph is None:
        best_S = S_cur
        best_graph = state.get_graph().copy()
    if sweep % 5 == 0 or sweep == 1:
        print(f"Sweep {sweep:2d}/{n_sweeps}: Entropy = {S_cur:8.4f} (dS = {dS:8.4f}, moves = {nmoves:4d})")


# ### Evaluation and Visualization
# Compare the inferred network with the ground truth topology both quantitatively and visually.

# In[75]:


def compute_reconstruction_metrics(W_true, W_inferred):
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)  # ignore diagonal
    
    gt_edges = (W_true[mask] != 0).astype(int)
    inf_edges = (W_inferred[mask] != 0).astype(int)
    
    tp = np.sum((gt_edges == 1) & (inf_edges == 1))
    fp = np.sum((gt_edges == 0) & (inf_edges == 1))
    fn = np.sum((gt_edges == 1) & (inf_edges == 0))
    tn = np.sum((gt_edges == 0) & (inf_edges == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    shd = fp + fn
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "TPR (Recall)": tpr, "FPR": fpr,
        "Precision": precision, "F1-Score": f1,
        "SHD": shd
    }

W_inferred = np.zeros((N, N), dtype=float)
try:
    weight_ep = state.x
except AttributeError:
    ep_keys = list(best_graph.ep.keys())
    weight_ep = best_graph.ep[ep_keys[0]] if ep_keys else None

for e in best_graph.edges():
    src, tgt = int(e.source()), int(e.target())
    if src != tgt:
        w = 1.0
        if weight_ep is not None:
            val = weight_ep[e]
            if hasattr(val, '__iter__') and not isinstance(val, str):
                w = float(val[0])
            else:
                w = float(val)
        W_inferred[tgt, src] = w

metrics = compute_reconstruction_metrics(W_true, W_inferred)
print("Reconstruction Metrics vs Ground Truth:")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")


# In[76]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original Matrix
im1 = axes[0].imshow(W_true, cmap='coolwarm', vmin=-3, vmax=3)
axes[0].set_title("Original GRN (W_true)")
axes[0].set_xlabel("Source Gene")
axes[0].set_ylabel("Target Gene")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Inferred Matrix
im2 = axes[1].imshow(W_inferred, cmap='coolwarm', vmin=-3, vmax=3)
axes[1].set_title("Inferred GRN (W_inferred)")
axes[1].set_xlabel("Source Gene")
axes[1].set_ylabel("Target Gene")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

