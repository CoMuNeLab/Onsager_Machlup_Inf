import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.inference.reconstruction_metrics import evaluate_topology, evaluate_weights

def main():
    parser = argparse.ArgumentParser(description="Compute metrics across grid search runs.")
    default_dir = os.path.join(PROJECT_ROOT, "results", "grid_results")
    parser.add_argument("--results_dir", type=str, default=default_dir, help="Directory containing trial folders")
    args = parser.parse_args()
    results_dir = args.results_dir
    
    records = []
    
    for trial_dir in glob.glob(os.path.join(results_dir, "*_trial_*")):
        dirname = os.path.basename(trial_dir)
        npz_path = os.path.join(trial_dir, "results.npz")
        
        if not os.path.exists(npz_path):
            continue
            
        parts = dirname.split("_")
        params = {"N": None, "m": 1, "p": None, "M": None, "dt": None, "twnd_start": None, "twnd_end": None, "trial": None}
        
        try:
            for i in range(len(parts)):
                if parts[i] == "N": params["N"] = int(parts[i+1])
                elif parts[i] == "m": params["m"] = int(parts[i+1])
                elif parts[i] == "p": params["p"] = float(parts[i+1])
                elif parts[i] == "M": params["M"] = int(parts[i+1])
                elif parts[i] == "dt": params["dt"] = float(parts[i+1])
                elif parts[i] == "twnd": 
                    params["twnd_start"] = float(parts[i+1])
                    params["twnd_end"] = float(parts[i+2])
                elif parts[i] == "trial": params["trial"] = int(parts[i+1])
                
            N = params["N"]
            m = params["m"]
            p = params["p"]
            M = params["M"]
            dt = params["dt"]
            twnd_start = params["twnd_start"]
            twnd_end = params["twnd_end"]
            trial = params["trial"]
            
            if N is None or M is None or dt is None or twnd_start is None:
                continue
        except (IndexError, ValueError):
            continue
            
        seed_network = 42 + trial
        if p is not None:
            # It's an ER network
            from inference_pipeline.data_generation.network_generator import generate_erdos_renyi_W
            W_true = generate_erdos_renyi_W(N, p=p, seed=seed_network)
        else:
            # It's a Scale-Free network
            W_true = generate_scale_free_W(N, m=m, seed=seed_network)
        
        data = np.load(npz_path)
        W_ensemble = data['W_ensemble']
        edge_probabilities = data['edge_probabilities']
        
        # Evaluate topology (binary prediction using the new functions)
        top_metrics = evaluate_topology(W_true, W_ensemble, threshold=0.0, weighted=False)
        
        # Evaluate weights
        weight_metrics = evaluate_weights(W_true, W_ensemble, threshold=0.0)
        
        # Compute classic binary metrics (TP, FP, FN, TN, etc.) using a 0.5 threshold on edge probabilities
        pred_binary = (edge_probabilities > 0.5).astype(int)
        true_binary = (np.abs(W_true) > 0).astype(int)
        
        # Exclude self-loops
        mask = ~np.eye(N, dtype=bool)
        pred_binary = pred_binary[mask]
        true_binary = true_binary[mask]
        
        TP = np.sum((pred_binary == 1) & (true_binary == 1))
        FP = np.sum((pred_binary == 1) & (true_binary == 0))
        FN = np.sum((pred_binary == 0) & (true_binary == 1))
        TN = np.sum((pred_binary == 0) & (true_binary == 0))
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1_Score = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0
        SHD = FP + FN
        
        regime = "Transient" if twnd_start == 0.0 else "Stationary"
        
        record = {
            "trial": trial,
            "dt_meas": dt,
            "M_pairs": M,
            "t_window_start": twnd_start,
            "t_window_end": twnd_end,
            "N": N,
            "m": m,
            "Regime": regime,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "TPR (Recall)": TPR,
            "FPR": FPR,
            "Precision": Precision,
            "F1-Score": F1_Score,
            "SHD": SHD,
            "AUROC": top_metrics.get("auroc", np.nan),
            "AUPRC": top_metrics.get("auprc", np.nan),
            "AUPRC_Ratio": top_metrics.get("auprc_ratio", np.nan),
            "AUPRC_Norm": top_metrics.get("auprc_norm", np.nan),
            "MSE": weight_metrics.get("mse", np.nan),
            "Pearson": weight_metrics.get("pearson", np.nan),
            "Spearman": weight_metrics.get("spearman", np.nan),
            "RBO": weight_metrics.get("rbo", np.nan),
            "Wasserstein": weight_metrics.get("wasserstein", np.nan)
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No valid result files found.")
        return
        
    out_path = os.path.join(results_dir, "performance_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully computed metrics and saved to {out_path}")

if __name__ == "__main__":
    main()
