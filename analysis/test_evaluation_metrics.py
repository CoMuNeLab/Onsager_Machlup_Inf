import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reconstruction_metrics import evaluate_topology, evaluate_weights
from perturbation_metrics import evaluate_shift_correlation, evaluate_calibration, evaluate_multimodal_kde

def run_tests():
    np.random.seed(42)
    
    print("Testing Reconstruction Metrics...")
    # N=10 nodes, M=100 samples
    N = 10
    M = 100
    
    # Ground truth: 20 edges
    W_true = np.zeros((N, N))
    edges = np.random.choice(N*N, 20, replace=False)
    W_true.flat[edges] = np.random.randn(20) * 5
    
    # Ensemble: noisy version
    W_ensemble = np.zeros((M, N, N))
    for m in range(M):
        # Keep 80% of true edges, add 10% false edges
        m_edges = np.zeros((N, N))
        for idx in edges:
            if np.random.rand() < 0.8:
                m_edges.flat[idx] = W_true.flat[idx] + (np.random.randn() * 0.5)
        
        false_edges = np.random.choice(N*N, 10, replace=False)
        for idx in false_edges:
             if idx not in edges:
                 m_edges.flat[idx] = np.random.randn() * 0.2
                 
        W_ensemble[m] = m_edges
        
    top_res = evaluate_topology(W_true, W_ensemble)
    print(f"Topology: {top_res}")
    assert "auroc" in top_res and not np.isnan(top_res["auroc"])
    
    top_res_w = evaluate_topology(W_true, W_ensemble, weighted=True)
    print(f"Topology (Weighted): {top_res_w}")
    
    weight_res = evaluate_weights(W_true, W_ensemble)
    print(f"Weights: {weight_res}")
    assert "wasserstein" in weight_res and not np.isnan(weight_res["wasserstein"])
    assert "rbo" in weight_res and not np.isnan(weight_res["rbo"])
    
    print("\nTesting Perturbation Metrics...")
    true_wt = np.random.rand(N)
    true_pert = true_wt + np.random.randn(N)
    true_pert[0] = 0.0 # ko_idx = 0
    
    ensemble_wt = np.random.rand(M, N)
    # Add noise to true shifts
    ensemble_shifts = (true_pert - true_wt) + np.random.randn(M, N) * 0.5
    ensemble_pert = ensemble_wt + ensemble_shifts
    ensemble_pert[:, 0] = 0.0
    
    shift_res = evaluate_shift_correlation(true_wt, true_pert, ensemble_wt, ensemble_pert, ko_idx=0)
    print(f"Shift Correlation: {shift_res}")
    assert "spearman_shift" in shift_res
    
    calib_res = evaluate_calibration(true_pert, ensemble_pert, ko_idx=0)
    print(f"Calibration: {calib_res}")
    
    kde_res = evaluate_multimodal_kde(true_pert, ensemble_pert, ko_idx=0)
    print(f"KDE Log-Likelihood: {kde_res}")
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    run_tests()
