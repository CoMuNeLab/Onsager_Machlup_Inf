import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import sys

nb = new_notebook()

nb.cells.append(new_markdown_cell("""\
# Wilson-Cowan Neural Network Simulation and Inference

This notebook demonstrates the robustness of the inference method to model-mismatch by generating synthetic data from a highly non-linear neural rate model (Wilson-Cowan) and successfully inferring its topology using the SDE-based global parameter MCMC engine.
"""))

nb.cells.append(new_code_cell("""\
import numpy as np
import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import sys
import os

sys.path.append(os.path.abspath('.'))
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import compute_reconstruction_metrics
from gene_expr import GeneExprBlockState
"""))

nb.cells.append(new_markdown_cell("""\
## 1. Simulate the Wilson-Cowan Network

We generate a 10-node directed Erdos-Renyi graph and simulate its dynamics using the Wilson-Cowan ODEs. We use 20 independent trials to simulate transient responses to stimuli, providing the inference algorithm with a rich dynamical dataset.
"""))

nb.cells.append(new_code_cell("""\
np.random.seed(42)
N = 10
p_edge = 0.3

# Generate Ground Truth Graph
G_true = nx.erdos_renyi_graph(n=N, p=p_edge, directed=True)
W_true = nx.to_numpy_array(G_true)
# Randomly assign +3.0 (excitatory) or -3.0 (inhibitory) to edges
signs = np.random.choice([-1.0, 1.0], size=(N, N))
W_true = W_true * signs * 3.0

# Simulation parameters
tau = 2.0         
I = -1.5          
sigma = 0.05      # Low noise to preserve signal
dt_sim = 0.01
steps_per_trial = 500  # 5 time units per trial
num_trials = 20        # 20 different initial conditions
subsample = 10         # dt_infer = 0.1

print(f"Simulating Wilson-Cowan Neural Network (N={N})...")
pairs = []
trajectories = []

for trial in range(num_trials):
    x = np.random.uniform(0.1, 0.9, size=N)
    trial_traj = []
    
    for step in range(steps_per_trial):
        net_input = W_true @ x + I
        activation = expit(net_input)
        dx = (-x / tau + (1 - x) * activation) * dt_sim
        noise = sigma * np.sqrt(dt_sim) * np.random.randn(N)
        
        x = x + dx + noise
        x = np.clip(x, 0.01, 0.99)
        
        if step % subsample == 0:
            trial_traj.append(x.copy())
            
    # Add pairs for this trial
    for i in range(len(trial_traj) - 1):
        pairs.append(np.column_stack([trial_traj[i], trial_traj[i+1]]))
    trajectories.extend(trial_traj)

trajectories = np.array(trajectories)
dt_infer = dt_sim * subsample
print(f"Generated {len(pairs)} transition pairs with dt={dt_infer}")

# Plot transients
plt.figure(figsize=(10, 4))
plt.plot(trajectories[:200]) # Plot first few trials
plt.title('Wilson-Cowan Transients (First 2 Trials)')
plt.xlabel('Time Step')
plt.ylabel('Firing Rate')
plt.show()
"""))

nb.cells.append(new_markdown_cell("""\
## 2. Run MCMC Inference

We now feed the dataset into the standard `GeneExprBlockState` engine. Even though the generating physics differ significantly from the internal SDE solver, we expect the engine to capture the primary topological constraints.
"""))

nb.cells.append(new_code_cell("""\
g_init = gt.Graph(directed=True)
g_init.add_vertex(N)
g_init.add_edge(0, 1)

# Initial theta: [log(decay), log(V), log(noise)]
initial_theta = [np.log(1/tau), np.log(1.0), np.log(0.01)]

print("Initializing Inference Engine...")
state = GeneExprBlockState(pairs, dt=dt_infer, g=g_init, state_args=dict(deg_corr=True), theta=initial_theta)

print("Running Burn-in (100 sweeps)...")
state = run_burn_in(state, n_burnin=100, theta_mode="global")

print("Running Posterior Sampling (200 sweeps)...")
W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=200, delta_k=1, theta_mode="global")

edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
W_inferred = W_ensemble[-1]
"""))

nb.cells.append(new_markdown_cell("""\
## 3. Evaluation and Metrics

Finally, we calculate the F1-Score, True Positive Rate, and Structural Hamming Distance to evaluate topological reconstruction, and visualize the true vs. inferred adjacency matrices.
"""))

nb.cells.append(new_code_cell("""\
metrics = compute_reconstruction_metrics(W_true, W_inferred)
print("--- RESULTS ---")
print(f"F1-Score: {metrics['F1-Score']:.3f}")
print(f"TPR (Recall): {metrics['TPR (Recall)']:.3f} | FPR: {metrics['FPR']:.3f} | Precision: {metrics['Precision']:.3f}")
print(f"SHD: {metrics['SHD']}")

# Plot Probability Matrix vs True Adjacency
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap((W_true > 0).astype(int), ax=axes[0], cmap="Blues", cbar=False, square=True)
axes[0].set_title("True Neural Network Topology")
axes[0].set_xlabel("Target Node")
axes[0].set_ylabel("Source Node")

sns.heatmap(edge_probabilities, ax=axes[1], cmap="Reds", vmin=0, vmax=1.0, square=True)
axes[1].set_title("Inferred Edge Probabilities")
axes[1].set_xlabel("Target Node")
axes[1].set_ylabel("Source Node")

plt.tight_layout()
plt.show()
"""))

with open('Wilson_Cowan_Inference_Benchmark.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook created successfully.")
