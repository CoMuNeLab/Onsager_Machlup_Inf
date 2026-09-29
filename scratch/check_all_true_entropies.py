import numpy as np
import graph_tool.all as gt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from gene_expr import GeneExprBlockState

N = 10
m = 1
seed_network = 42
W_true = generate_scale_free_W(N, m, seed=seed_network)
dt_meas = 0.5

scenarios = [
    {"name": "Baseline (M=100, Noise=0.02)", "M_pairs": 100, "D_base": 0.02},
    {"name": "High Data (M=500, Noise=0.02)", "M_pairs": 500, "D_base": 0.02},
    {"name": "Low Noise (M=100, Noise=0.002)", "M_pairs": 100, "D_base": 0.002},
    {"name": "Ideal (M=500, Noise=0.002)", "M_pairs": 500, "D_base": 0.002}
]

for sc in scenarios:
    np.random.seed(42)
    gt.seed_rng(42)
    s, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
        W_true, M=sc["M_pairs"], dt_sim=0.005, dt_sample=0.5, dt_meas=dt_meas, 
        D_base=sc["D_base"], mean_d=0.5, std_d=0.5, mean_V=2.0, std_V=0.5, seed=123
    )

    g_true = gt.Graph(directed=True)
    g_true.add_vertex(N)
    for u in range(N):
        for v in range(N):
            if W_true[v, u] != 0:
                e = g_true.add_edge(u, v)

    state_true = GeneExprBlockState(s, dt=dt_meas, g=g_true, state_args=dict(deg_corr=False), use_em_likelihood=False)

    x_prop = state_true.x
    for e in g_true.edges():
        x_prop[e] = W_true[int(e.target()), int(e.source())]

    arr = state_true.theta.get_2d_array([0, 1, 2])
    arr[0, :] = np.log(d_true)
    arr[1, :] = np.log(V_true)
    arr[2, :] = np.log(sc["D_base"])
    state_true.theta.set_2d_array(arr)

    print(f"{sc['name']} - True Entropy: {state_true.entropy():.4f}")
