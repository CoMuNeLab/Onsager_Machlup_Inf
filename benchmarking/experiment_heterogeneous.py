import numpy as np
import graph_tool.all as gt
import sys
import os

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from inference_pipeline.inference.metrics import compute_reconstruction_metrics
from gene_expr import GeneExprBlockState

def run_experiment():
    N = 10
    m = 1
    seed_network = 42

    dt_sim = 0.005
    dt_sample = 0.5
    dt_meas = 0.5
    M_pairs = 100
    t_window = (0.0, 3.0)

    mean_d = 0.5
    std_d = 0.5
    mean_V = 2.0
    std_V = 0.5
    D_base = 0.02

    print("Generating TRUE network...")
    W_true = generate_scale_free_W(N, m, seed=seed_network)

    scenarios = [
        {"name": "Homogeneous Data + Global Mode", "std_d": 0.0, "std_V": 0.0, "mode": "global"},
        {"name": "Heterogeneous Data + Local Mode", "std_d": 0.5, "std_V": 0.5, "mode": "local"}
    ]

    results = {}

    for sc in scenarios:
        print(f"\n=========================================")
        print(f"Scenario: {sc['name']}")
        print(f"=========================================")
        
        # Reset seed for reproducibility between scenarios
        np.random.seed(42)
        gt.seed_rng(42)
        
        # Generate data specific to the scenario
        s, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
            W_true, M=M_pairs, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
            D_base=D_base, mean_d=mean_d, std_d=sc["std_d"], mean_V=mean_V, std_V=sc["std_V"], 
            seed=123, t_window=t_window
        )
        print(f"Harvested {len(s)} pairs. Master Regulator is Node {hub_node}.")

        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)

        num_initial_edges = max(1, int(N * N * 0.05))
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:num_initial_edges]:
            g_init.add_edge(u, v)

        initial_theta = [0.0, 0.0, np.log(0.05)]
        
        state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=False), theta=initial_theta, use_em_likelihood=False)
        print(f"Initial description length (empty network): {state.entropy():.4f}")

        state = run_burn_in(state, n_burnin=50, theta_mode=sc["mode"])
        
        W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, theta_mode=sc["mode"])
        
        metrics = compute_reconstruction_metrics(W_true, W_ensemble[-1])
        results[sc["name"]] = metrics
        
        print(f"\nMetrics for {sc['name']}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    print("\n================ SUMMARY ================")
    for sc in scenarios:
        name = sc["name"]
        print(f"Scenario: {name}")
        if 'AUROC' in results[name]:
            print(f"  AUROC: {results[name]['AUROC']:.4f}")
        if 'AUPRC' in results[name]:
            print(f"  AUPRC: {results[name]['AUPRC']:.4f}")
        if 'MCC' in results[name]:
            print(f"  MCC:   {results[name]['MCC']:.4f}")
        print(f"  F1-Score: {results[name].get('F1-Score', 0):.4f}")
        print(f"  SHD: {results[name].get('SHD', 0)}")


if __name__ == "__main__":
    run_experiment()
