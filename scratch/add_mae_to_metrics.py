import os
import numpy as np
import pandas as pd
from inference_pipeline.data_generation.network_generator import generate_scale_free_W

def main():
    csv_path = "results/grid_results/performance_metrics.csv"
    df = pd.read_csv(csv_path)
    results_dir = "results/grid_results"
    
    maes = []
    
    for idx, row in df.iterrows():
        if row['status'] != 'success':
            maes.append(np.nan)
            continue
            
        N = int(row['N'])
        M = int(row['M_pairs'])
        dt = float(row['dt_meas'])
        twnd_start = float(row['t_window_start'])
        twnd_end = float(row['t_window_end'])
        trial = int(row['trial'])
        
        config_name = f"N_{N}_M_{M}_dt_{dt}_twnd_{twnd_start}_{twnd_end}_trial_{trial}"
        npz_path = os.path.join(results_dir, config_name, "results.npz")
        
        if not os.path.exists(npz_path):
            maes.append(np.nan)
            continue
            
        seed_network = 42 + trial
        m = 1
        W_true = generate_scale_free_W(N, m, seed=seed_network)
        
        data = np.load(npz_path)
        weight_mean = data['weight_mean']
        
        # Exclude diagonal
        mask = ~np.eye(N, dtype=bool)
        mae = np.mean(np.abs(W_true[mask] - weight_mean[mask]))
        maes.append(mae)
        
    df['Weighted_MAE'] = maes
    df.to_csv(csv_path, index=False)
    print("Successfully added Weighted_MAE to performance_metrics.csv")

if __name__ == "__main__":
    main()
