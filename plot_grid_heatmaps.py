import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

from inference_pipeline.data_generation.network_generator import generate_scale_free_W

def compute_metrics(true_binary, pred_prob, W_true, weight_mean):
    y_true = true_binary.flatten()
    y_scores = pred_prob.flatten()
    # Remove diagonal elements (self-loops)
    mask = ~np.eye(true_binary.shape[0], dtype=bool).flatten()
    try:
        auc = roc_auc_score(y_true[mask], y_scores[mask])
        ap = average_precision_score(y_true[mask], y_scores[mask])
    except ValueError:
        auc = np.nan
        ap = np.nan
        
    mask_2d = ~np.eye(true_binary.shape[0], dtype=bool)
    mae = np.mean(np.abs(W_true[mask_2d] - weight_mean[mask_2d]))
        
    return auc, ap, mae

def main():
    results_dir = "results/grid_results"
    
    # 1. Collect and Compute Metrics
    print("Gathering data and computing AUROC/AUPRC...")
    records = []
    
    for trial_dir in glob.glob(os.path.join(results_dir, "*_trial_*")):
        dirname = os.path.basename(trial_dir)
        npz_path = os.path.join(trial_dir, "results.npz")
        
        if not os.path.exists(npz_path):
            continue
            
        # Parse directory name: N_20_M_500_dt_0.05_twnd_0.0_3.0_trial_0
        parts = dirname.split("_")
        try:
            N = int(parts[1])
            M = int(parts[3])
            dt = float(parts[5])
            twnd_start = float(parts[7])
            twnd_end = float(parts[8])
            trial = int(parts[10])
        except (IndexError, ValueError):
            continue
            
        # Re-generate true network to get binary labels
        seed_network = 42 + trial
        m = 1
        W_true = generate_scale_free_W(N, m, seed=seed_network)
        W_true_binary = (W_true != 0).astype(int)
        
        # Load predicted edge probabilities
        data = np.load(npz_path)
        edge_probabilities = data['edge_probabilities']
        weight_mean = data['weight_mean']
        
        auroc, auprc, mae = compute_metrics(W_true_binary, edge_probabilities, W_true, weight_mean)
        
        regime = "Transient" if twnd_start == 0.0 else "Stationary"
        
        records.append({
            "N": N,
            "M_pairs": M,
            "dt_meas": dt,
            "Regime": regime,
            "trial": trial,
            "AUROC": auroc,
            "AUPRC": auprc,
            "Weighted_MAE": mae
        })
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No valid result files found.")
        return
        
    # Aggregate over trials
    df.to_csv(os.path.join(results_dir, "performance_metrics_extended_raw.csv"), index=False)
    df_agg = df.groupby(["N", "M_pairs", "dt_meas", "Regime"]).mean().reset_index()
    df_agg.to_csv(os.path.join(results_dir, "performance_metrics_extended.csv"), index=False)
    
    # Create Output Directory
    out_dir = "images/heatmaps"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating heatmaps...")
    
    sns.set_theme(style="white", context="talk")
    
    from scipy.interpolate import RectBivariateSpline
    import matplotlib.ticker as ticker
    
    # Function to plot heatmaps
    def plot_heatmap(data, metric, title, filename, cmap, vmin=0.5, vmax=1.0):
        # Pivot the dataframe for the heatmap
        # Rows: dt_meas, Cols: M_pairs
        pivot = data.pivot(index="dt_meas", columns="M_pairs", values=metric)
        # Ensure rows are sorted nicely (ascending for interpolation)
        pivot = pivot.sort_index(ascending=True) 
        
        # Fill any NaNs
        pivot_filled = pivot.ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0).fillna(0)
        
        X = pivot_filled.columns.values.astype(float)
        Y = pivot_filled.index.values.astype(float)
        Z = pivot_filled.values
        
        # We interpolate in log-space for coordinates so it looks natural on log-scaled axes
        X_log = np.log10(X)
        Y_log = np.log10(Y)
        
        # Bivariate spline requires kx, ky < length of dimensions
        kx = min(3, len(X)-1)
        ky = min(3, len(Y)-1)
        
        spline = RectBivariateSpline(Y_log, X_log, Z, kx=ky, ky=kx)
        
        Y_fine_log = np.linspace(Y_log.min(), Y_log.max(), 100)
        X_fine_log = np.linspace(X_log.min(), X_log.max(), 100)
        
        Z_fine = spline(Y_fine_log, X_fine_log)
        # Clip to restrict color scale and prevent overshoot
        if vmin is not None and vmax is not None:
            Z_fine = np.clip(Z_fine, vmin, vmax)
            levels = np.linspace(vmin, vmax, 51)
        else:
            levels = 50
        
        X_fine = np.power(10, X_fine_log)
        Y_fine = np.power(10, Y_fine_log)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        c = ax.contourf(X_fine, Y_fine, Z_fine, levels=levels, cmap=cmap, extend='both' if vmin is not None else 'neither')
        
        fig.colorbar(c, ax=ax)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Show exact original ticks
        ax.set_xticks(X)
        ax.set_yticks(Y)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
        ax.set_title(title, pad=20)
        ax.set_ylabel(r"Measurement Interval ($\Delta t$)")
        ax.set_xlabel("Number of Samples ($M$)")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
    # Generate a plot for each combination of N and Regime
    for n_val in df_agg["N"].unique():
        for reg in df_agg["Regime"].unique():
            subset = df_agg[(df_agg["N"] == n_val) & (df_agg["Regime"] == reg)]
            if len(subset) == 0:
                continue
            
            plot_heatmap(
                subset, "AUPRC", 
                f"AUPRC (N={n_val}, {reg} Regime)",
                os.path.join(out_dir, f"heatmap_AUPRC_N{n_val}_{reg}.png"),
                "magma", vmin=0.5, vmax=1.0
            )
            plot_heatmap(
                subset, "AUROC", 
                f"AUROC (N={n_val}, {reg} Regime)",
                os.path.join(out_dir, f"heatmap_AUROC_N{n_val}_{reg}.png"),
                "viridis", vmin=0.5, vmax=1.0
            )
            plot_heatmap(
                subset, "Weighted_MAE", 
                f"Weighted MAE (N={n_val}, {reg} Regime)",
                os.path.join(out_dir, f"heatmap_MAE_N{n_val}_{reg}.png"),
                "coolwarm", vmin=None, vmax=None
            )
            
    print(f"Heatmaps successfully saved in {out_dir}/")

if __name__ == "__main__":
    main()
