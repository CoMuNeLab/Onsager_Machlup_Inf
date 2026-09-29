import numpy as np
import graph_tool.all as gt
import sys
import os
import json
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling
from inference_pipeline.inference.metrics import compute_reconstruction_metrics
from gene_expr import GeneExprBlockState

def compute_rbo(l1, l2, p=0.9):
    if not l1 and not l2: return 1.0
    if not l1 or not l2: return 0.0
    set1, set2 = set(), set()
    score = 0.0
    for d, (e1, e2) in enumerate(zip(l1, l2), start=1):
        set1.add(e1)
        set2.add(e2)
        overlap = len(set1.intersection(set2))
        score += (overlap / d) * (p ** (d-1))
    return score * (1 - p)

def compute_extended_metrics(W_true, W_ensemble):
    edge_exists = (W_ensemble != 0).astype(float)
    edge_probabilities = np.mean(edge_exists, axis=0)
    
    masked_W = np.ma.masked_equal(W_ensemble, 0)
    weight_mean = masked_W.mean(axis=0).filled(0.0)
    
    metrics = compute_reconstruction_metrics(W_true, W_ensemble[-1])
    
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)
    
    y_true = (W_true[mask] != 0).astype(int)
    y_scores = edge_probabilities[mask]
    
    if len(np.unique(y_true)) > 1:
        metrics['AUROC'] = roc_auc_score(y_true, y_scores)
        metrics['AUPRC'] = average_precision_score(y_true, y_scores)
    else:
        metrics['AUROC'] = np.nan
        metrics['AUPRC'] = np.nan
        
    metrics['MAE'] = mean_absolute_error(W_true[mask], weight_mean[mask])
    
    true_edges = []
    inf_edges = []
    for i in range(N):
        for j in range(N):
            if i != j:
                true_edges.append(((i,j), abs(W_true[i,j])))
                inf_edges.append(((i,j), edge_probabilities[i,j] * abs(weight_mean[i,j])))
                
    true_edges.sort(key=lambda x: x[1], reverse=True)
    inf_edges.sort(key=lambda x: x[1], reverse=True)
    
    k = 2*N
    l1 = [e[0] for e in true_edges[:k]]
    l2 = [e[0] for e in inf_edges[:k]]
    metrics['RBO'] = compute_rbo(l1, l2)
    
    for key, val in metrics.items():
        if isinstance(val, (np.integer, np.int64, np.int32)):
            metrics[key] = int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            metrics[key] = float(val)
            
    return metrics

def calculate_true_entropy(s, W_true, d_true, V_true, D_base, dt_meas):
    N = W_true.shape[0]
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
    arr[2, :] = np.log(D_base)
    state_true.theta.set_2d_array(arr)

    return state_true.entropy()

def main():
    networks = [10, 20]
    dt_sim = 0.005
    dt_sample = 0.5
    dt_meas = 0.5
    t_window = (0.0, 3.0)
    M_pairs = 500
    D_base = 0.02
    num_trials = 3

    stds = [0.1, 0.5, 1.0, 1.5, 2.0]

    all_results = {}

    for N in networks:
        print(f"\n=========================================")
        print(f"Network Size: N={N}")
        print(f"=========================================")
        all_results[N] = {}
        
        for std in stds:
            scenario_name = f"std_{std:.1f}"
            print(f"\n--- Scenario: {scenario_name} ---")
            all_results[N][scenario_name] = []
            
            for trial in range(num_trials):
                # Ensure each trial uses a different random network and data
                seed_network = 42 + trial * 10
                seed_data = 123 + trial * 10
                
                print(f"  Trial {trial+1}/{num_trials}")
                np.random.seed(seed_network)
                gt.seed_rng(seed_network)
                
                W_true = generate_scale_free_W(N, m=1 if N==10 else 2, seed=seed_network)
                
                s, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
                    W_true, M=M_pairs, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
                    D_base=D_base, mean_d=0.5, std_d=std, mean_V=2.0, std_V=std, 
                    seed=seed_data, t_window=t_window
                )

                true_ent = calculate_true_entropy(s, W_true, d_true, V_true, D_base, dt_meas)
                
                g_init = gt.Graph(directed=True)
                g_init.add_vertex(N)
                num_initial_edges = max(1, int(N * N * 0.05))
                possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
                np.random.shuffle(possible_edges)
                for u, v in possible_edges[:num_initial_edges]:
                    g_init.add_edge(u, v)

                initial_theta = [0.0, 0.0, np.log(0.05)]
                state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=False), theta=initial_theta, use_em_likelihood=False)
                
                state = run_burn_in(state, n_burnin=50, theta_mode="local")
                W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, theta_mode="local")
                
                inferred_ent = state.entropy()
                metrics = compute_extended_metrics(W_true, W_ensemble)
                metrics["True Entropy"] = true_ent
                metrics["Inferred Entropy"] = inferred_ent
                
                
                all_results[N][scenario_name].append(metrics)
                
                # Save incrementally to prevent data loss
                os.makedirs("results", exist_ok=True)
                with open("results/high_variance_benchmark.json", "w") as f:
                    json.dump(all_results, f, indent=4)
                
    print("\nBenchmark completed and saved to results/high_variance_benchmark.json")

if __name__ == "__main__":
    main()
