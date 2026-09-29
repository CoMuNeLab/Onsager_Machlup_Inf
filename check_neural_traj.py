import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import expit
import os

np.random.seed(42)
N = 10
p_edge = 0.3
G_true = nx.erdos_renyi_graph(n=N, p=p_edge, directed=True)
W_true = nx.to_numpy_array(G_true)
W_true *= 5.0 

tau = 2.0
I = -2.0
sigma = 0.1
dt_sim = 0.01
T_steps = 100000
subsample = 100

x = np.random.uniform(0.1, 0.9, size=N)
trajectories = []

for step in range(T_steps):
    net_input = W_true @ x + I
    activation = expit(net_input)
    dx = (-x / tau + (1 - x) * activation) * dt_sim
    noise = sigma * np.sqrt(dt_sim) * np.random.randn(N)
    
    x = x + dx + noise
    x = np.clip(x, 0.01, 0.99)
    
    if step % subsample == 0:
        trajectories.append(x.copy())

trajectories = np.array(trajectories)
plt.plot(trajectories)
plt.title('Wilson-Cowan Trajectories (Subsampled)')
plt.savefig('neural_traj.png')
print(f"Traj Mean: {trajectories.mean():.3f}, Std: {trajectories.std():.3f}")
