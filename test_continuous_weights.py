import numpy as np
import networkx as nx
import graph_tool.all as gt
from gene_expr import GeneExprBlockState
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling
from inference_pipeline.inference.metrics import compute_reconstruction_metrics
from scipy.special import expit
import sys
import os

N = 10
p_edge = 0.3
np.random.seed(42)

G_true = nx.erdos_renyi_graph(n=N, p=p_edge, directed=True)
W_base = nx.to_numpy_array(G_true)
signs = np.random.choice([-1, 1], size=(N, N))
# CONTINUOUS TRUE WEIGHTS (Normal dist around +/-3.0)
W_true = W_base * signs * np.random.normal(3.0, 1.0, size=(N,N))

tau = 2.0
I = -1.5
sigma = 0.05
dt_sim = 0.01
steps = 500
trials = 20
subsample = 10
dt_infer = dt_sim * subsample

pairs = []
for trial in range(trials):
    x = np.random.uniform(0.1, 0.9, size=N)
    trial_traj = []
    for step in range(steps):
        activation = expit(W_true @ x + I)
        dx = (-x/tau + (1-x)*activation)*dt_sim
        noise = sigma*np.sqrt(dt_sim)*np.random.randn(N)
        x = np.clip(x + dx + noise, 0.01, 0.99)
        if step % subsample == 0:
            trial_traj.append(x.copy())
    for i in range(len(trial_traj)-1):
        pairs.append(np.column_stack([trial_traj[i], trial_traj[i+1]]))

g_init = gt.Graph(directed=True)
g_init.add_vertex(N)
g_init.add_edge(0, 1)

state = GeneExprBlockState(pairs, dt=dt_infer, g=g_init, theta=[np.log(0.5), np.log(1.0), np.log(0.01)])
state = run_burn_in(state, n_burnin=50)
W_ens, _ = run_posterior_sampling(state, N, n_samples=100)

metrics = compute_reconstruction_metrics(W_true, W_ens[-1])
print("METRICS:", metrics)
