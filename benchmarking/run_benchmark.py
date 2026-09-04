import os
import sys
import glob
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

import methods.method_graph_tool as m_gt
import methods.method_random_forest as m_rf
import methods.method_sparse_ode as m_ode
import methods.method_correlation as m_corr

def evaluate_predictions(edge_scores, W_true):
    """
    Evaluates the continuous edge scores against the binary ground-truth graph.
    Returns AUPRC and AUROC.
    """
    N = W_true.shape[0]
    mask = ~np.eye(N, dtype=bool)  # Ignore self-loops
    
    y_true = (W_true[mask] != 0).astype(int)
    y_score = edge_scores[mask]
    
    # AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.5  # If only one class is present in y_true, AUROC is undefined
        
    return auprc, auroc

def run_benchmark():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    
    dataset_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npz"):
                dataset_files.append(os.path.join(root, file))
                
    dataset_files.sort()
    
    algorithms = {
        "Graph-Tool (SDE)": m_gt.infer_network,
        "Random Forest (dynGENIE3-like)": m_rf.infer_network,
        "Sparse ODE (Lasso)": m_ode.infer_network,
        "Time-Delayed Correlation": m_corr.infer_network
    }
    
    for ds_file in dataset_files:
        print(f"\n========================================")
        print(f"Evaluating dataset: {os.path.basename(ds_file)}")
        print(f"========================================")
        
        data = np.load(ds_file, allow_pickle=True)
        X = data["X"]
        t = data["t"]
        W_true = data["W_true"]
        
        for algo_name, infer_func in algorithms.items():
            print(f"Running {algo_name}...")
            
            # For Graph-Tool, we use fewer sweeps in benchmarking for speed, 
            # unless specified otherwise in the wrapper.
            edge_scores = infer_func(X, t)
            
            auprc, auroc = evaluate_predictions(edge_scores, W_true)
            
            print(f"  -> AUPRC: {auprc:.4f} | AUROC: {auroc:.4f}")

if __name__ == "__main__":
    run_benchmark()
