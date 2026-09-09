import os
import numpy as np
import graph_tool.all as gt

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
from gene_expr import GeneExprBlockState
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(true_binary, pred_prob):
    y_true = true_binary.flatten()
    y_scores = pred_prob.flatten()
    mask = ~np.eye(true_binary.shape[0], dtype=bool).flatten()
    auc = roc_auc_score(y_true[mask], y_scores[mask])
    ap = average_precision_score(y_true[mask], y_scores[mask])
    return auc, ap

def run_experiment():
    os.makedirs("results/model_exp", exist_ok=True)
    
    # Simulation settings
    N = 20
    M = 500
    dt_meas = 1.0
    m = 1
    seed_network = 48
    
    dt_sim = 0.005
    dt_sample = 1.0
    mean_d = 0.5
    mean_V = 2.0
    D_base = 0.02
    t_window = (0.0, 5.1)
    
    # 1. Generate Data
    W_true = generate_scale_free_W(N, m, seed=seed_network)
    W_true_binary = (W_true != 0).astype(int)
    
    s, hub_node, out_degrees, d_true, V_true = simulate_realistic_pairs(
        W_true, M=M, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
        D_base=D_base, mean_d=mean_d, std_d=0.0, mean_V=mean_V, std_V=0.0, 
        seed=123, t_window=t_window
    )
    
    def run_inference(setting_name, use_em_likelihood, theta_mode):
        print(f"\n{'='*50}")
        print(f"Running Inference Setting: {setting_name}")
        print(f"{'='*50}")
        
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)
        
        num_initial_edges = max(1, int(N * N * 0.05))
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:num_initial_edges]:
            g_init.add_edge(u, v)
            
        initial_theta = [0.0, 0.0, np.log(0.05)]
        
        state = GeneExprBlockState(s, dt=dt_meas, g=g_init, state_args=dict(deg_corr=True), 
                                   theta=initial_theta, use_em_likelihood=use_em_likelihood)
        
        state = run_burn_in(state, n_burnin=50, theta_mode=theta_mode)
        W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, theta_mode=theta_mode)
        
        edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
        weight_last = W_ensemble[-1]
        return edge_probabilities, weight_last

    # Run the three settings
    res_1_prob, res_1_weight_last = run_inference("1_Analytical_Global", False, "global")
    res_2_prob, res_2_weight_last = run_inference("2_EulerMaruyama_Global", True, "global")
    res_3_prob, res_3_weight_last = run_inference("3_Analytical_Local", False, "local")
    
    print("\n--- SAVING RESULTS ---")
    np.savez(
        f"results/model_exp/results_N{N}_M{M}_dt{dt_meas*10:.0f}.npz",
        W_true=W_true,
        W_true_binary=W_true_binary,
        res_1_prob=res_1_prob,
        res_1_weight_last=res_1_weight_last,
        res_2_prob=res_2_prob,
        res_2_weight_last=res_2_weight_last,
        res_3_prob=res_3_prob,
        res_3_weight_last=res_3_weight_last
    )
    
    auc1, ap1 = compute_metrics(W_true_binary, res_1_prob)
    auc2, ap2 = compute_metrics(W_true_binary, res_2_prob)
    auc3, ap3 = compute_metrics(W_true_binary, res_3_prob)
    
    metrics_text = f"""
Performance Metrics:
1. Analytical Global: AUC = {auc1:.4f}, AP (AUPRC) = {ap1:.4f}
2. E-M Global:        AUC = {auc2:.4f}, AP (AUPRC) = {ap2:.4f}
3. Analytical Local:  AUC = {auc3:.4f}, AP (AUPRC) = {ap3:.4f}
"""
    with open(f"results/model_exp/metrics_N{N}_M{M}_dt{dt_meas*10:.0f}.txt", "w") as f:
        f.write(metrics_text)
        
    print("Inference completed and saved!")

if __name__ == "__main__":
    run_experiment()
