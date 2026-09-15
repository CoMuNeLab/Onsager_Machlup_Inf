#!/usr/bin/env python
# coding: utf-8

# # Gene Regulatory Network Inference Pipeline
# 
# This notebook orchestrates the entire GRN inference pipeline:
# 1. **Data Generation**: Generates a scale-free network and harvests time-series data pairs.
# 2. **Bayesian Inference**: Uses `graph-tool` and our custom `GeneExprBlockState` to run MCMC and infer the posterior ensemble of network topologies and parameters.
# 3. **Prediction**: Exports the ensemble to raw binaries and calls Julia to run high-performance stationary-state knockout simulations.

# In[1]:


import numpy as np
import graph_tool.all as gt
import sys
import os
import subprocess

# Add the parent directory to sys.path so we can import our modules and gene_expr


from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import plot_probabilistic_network, plot_inferred_vs_true_am, compute_reconstruction_metrics
from inference_pipeline.prediction.export_utils import export_ensemble_to_bin, export_true_to_bin
from gene_expr import GeneExprBlockState


# ## 1. Data Generation

# In[2]:


# Configuration
N = 10
m = 1
seed_network = 42

dt_sim = 0.005
dt_sample = 0.5
dt_meas = 0.5
M_pairs = 100
t_window = (0.0, 3.0)

mean_d = 0.5
mean_V = 2.0
D_base = 0.02

# Generate TRUE network
W_true = generate_scale_free_W(N, m, seed=seed_network)
theta_true = [np.log(mean_d), np.log(mean_V), np.log(D_base)]

# Simulate state-transition pairs
s, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
    W_true, M=M_pairs, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
    D_base=D_base, mean_d=mean_d, std_d=0.0, mean_V=mean_V, std_V=0.0, 
    seed=123, t_window=t_window
)

print(f"Harvested {len(s)} pairs. Master Regulator is Node {hub_node}.")


# ## 2. Bayesian Inference & MCMC

# In[3]:


# Instantiate an empty graph in graph-tool format
g_init = gt.Graph(directed=True)
g_init.add_vertex(N)

# Add ~5% density of random edges to jump-start the MCMC and avoid the empty graph trap
num_initial_edges = max(1, int(N * N * 0.05))
possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
np.random.shuffle(possible_edges)
for u, v in possible_edges[:num_initial_edges]:
    g_init.add_edge(u, v)

# Initialize the custom state with an uninformative prior
initial_theta = [0.0, 0.0, np.log(0.05)]
state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=False), theta=initial_theta, use_em_likelihood=False)
print(f"Initial description length (empty network): {state.entropy():.4f}")

# Phase 1: Burn-in
state = run_burn_in(state, n_burnin=50, theta_mode="global")

# Phase 2: Posterior Sampling
n_samples = 100
W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=n_samples, delta_k=1, theta_mode="global")

# Compute and save ensemble stats
edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)

import os
os.makedirs("results", exist_ok=True)
np.savez_compressed(
    "results/grn_ensemble_results.npz", 
    W_ensemble=W_ensemble, 
    theta_ensemble=theta_ensemble,
    edge_probabilities=edge_probabilities,
    weight_mean=weight_mean,
    weight_std=weight_std
)
print("Ensemble results saved to results/grn_ensemble_results.npz!")


# ### Inference Visualizations

# In[9]:


print("\n--- Inference Metrics (Last Sample) ---")
metrics = compute_reconstruction_metrics(W_true, W_ensemble[-1])
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")
print("---------------------------------------\n")

plot_probabilistic_network(W_true, edge_probabilities*np.abs(weight_mean), save_path="images/probabilistic_network.png")
plot_inferred_vs_true_am(W_true, np.mean(W_ensemble, axis=0), save_path="images/am_comparison.png")


# ## 3. Julia Perturbation Analysis

# In[ ]:


# Export numpy tensors to raw C-order binaries for fast Julia loading
export_ensemble_to_bin("results/grn_ensemble_results.npz", "results/W_ensemble.bin", "results/theta_ensemble.bin")
export_true_to_bin(W_true, theta_true, "results/W_true.bin", "results/theta_true.bin")

# Call the Julia script for knockout prediction and visualization
print(f"Running Julia simulation with Node {hub_node + 1}...")
subprocess.run(["julia", "prediction/predict_knockout.jl", str(hub_node + 1)], check=True)

