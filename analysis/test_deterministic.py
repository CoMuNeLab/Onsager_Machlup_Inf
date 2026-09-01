import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from analysis.exp_information_content import generate_network, simulate_trajectory

N = 20
topo = 'ER'
d = 0.5
V = 5.0
noise = 0.0

np.random.seed(42)
W = generate_network(N, topo)

tau = 1.0 / d
T_transient_end = 5.0 * tau
T_total = T_transient_end + 60.0

traj = simulate_trajectory(W, T_total=T_total, dt_sim=0.005, dt_meas=0.05, 
                           d=d, V=V, noise_std=noise)
                           
print(np.var(traj[int(T_transient_end/0.05):], axis=0))
