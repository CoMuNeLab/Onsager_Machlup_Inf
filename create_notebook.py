import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell("""# Custom GRN Topology Generation and Graph-Tool Inference

This notebook allows you to generate custom Gene Regulatory Networks (GRNs) using a directed Scale-Free topology, simulate their trajectories with heterogeneous kinetics, and perform Bayesian inference using the `gene_expr` extension for `graph-tool`."""))

nb.cells.append(new_code_cell("""import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in path to load gene_expr
if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))

import graph_tool.all as gt
from gene_expr import GeneExprBlockState"""))

nb.cells.append(new_code_cell("""def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))"""))

nb.cells.append(new_code_cell("""def generate_scale_free_W(N, m=2, seed=42):
    \"\"\"Generates a scale-free directed GRN using directed preferential attachment.\"\"\"
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
            
    return W"""))

nb.cells.append(new_code_cell("""def simulate_realistic_trajectories(W, T_total, dt, dt_meas, D_base, mean_d, std_d, mean_V, std_V, seed=123):
    N = W.shape[0]
    np.random.seed(seed)
    
    # Heterogeneous parameters (Log-Normal)
    d = np.random.lognormal(mean=np.log(mean_d), sigma=std_d, size=N)
    V = np.random.lognormal(mean=np.log(mean_V), sigma=std_V, size=N)
    
    n_steps = int(T_total / dt)
    x = np.random.uniform(0.1, 1.0, size=N)
    x_history = []
    t_history = []
    
    out_degrees = (W != 0).sum(axis=0)
    hub_node = np.argmax(out_degrees)
    
    meas_step = max(1, int(dt_meas / dt))
    
    for step in range(n_steps):
        t = step * dt
        drift = -d * x + V * sigmoid(W @ x)
        noise = np.sqrt(2 * D_base) * np.random.normal(0, np.sqrt(dt), size=N)
        
        x = x + drift * dt + noise
        x = np.maximum(x, 0.0)
        
        if step % meas_step == 0:
            x_history.append(x.copy())
            t_history.append(t)
            
    return np.array(t_history), np.array(x_history), hub_node, out_degrees"""))

nb.cells.append(new_markdown_cell("""### Custom Parameters & Trajectory Simulation
Set the parameters and generate the network and trajectories."""))

nb.cells.append(new_code_cell("""# Network Parameters
N = 10              # Number of genes
m = 2               # Edges to attach from a new node to existing nodes
seed_network = 42   # Random seed for network generation

# Simulation Parameters
T_total = 100.0     # Total simulation time
dt = 0.01           # Simulation time step
dt_meas = 0.5       # Measurement interval
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
t, X, hub_node, out_degrees = simulate_realistic_trajectories(
    W_true, T_total, dt, dt_meas, D_base, 
    mean_decay, std_decay, mean_V, std_V, seed=seed_sim
)

# Format for graph-tool inference (shape: N x M)
s = X.T
print(f"Master Regulator is Node {hub_node} with {out_degrees[hub_node]} outgoing edges.")
print(f"Trajectory shape (N x M): {s.shape}")"""))

nb.cells.append(new_markdown_cell("""### Graph-Tool Bayesian Inference
We now use the `GeneExprBlockState` from the graph-tool extension to infer the network topology from the time-series data."""))

nb.cells.append(new_code_cell("""print("Instantiating GeneExprBlockState...")
state = GeneExprBlockState(s, dt=dt_meas)
S_init = state.entropy()
print(f"Initial description length (entropy): {S_init:.4f}")

print("\\nRunning MCMC Sweeps (Network Structure + Parameters)...")
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
        print(f"Sweep {sweep:2d}/{n_sweeps}: Entropy = {S_cur:8.4f} (dS = {dS:8.4f}, moves = {nmoves:4d})")"""))

nb.cells.append(new_markdown_cell("""### Evaluation
Compare the inferred network with the ground truth topology."""))

nb.cells.append(new_code_cell("""def compute_reconstruction_metrics(W_true, W_inferred):
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

W_inferred = np.zeros((N, N), dtype=int)
for e in best_graph.edges():
    src, tgt = int(e.source()), int(e.target())
    if src != tgt:
        W_inferred[tgt, src] = 1

metrics = compute_reconstruction_metrics(W_true, W_inferred)
print("Reconstruction Metrics vs Ground Truth:")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")"""))

with open('custom_grn_inference.ipynb', 'w') as f:
    nbformat.write(nb, f)
