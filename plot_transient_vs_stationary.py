import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def label_window(row):
    if row['Regime'] == 'Transient':
        return 'T'
    else:
        return 'S'

def main():
    parser = argparse.ArgumentParser(description="Generate AUPRC Transient vs Stationary bar plots.")
    parser.add_argument('--input', type=str, default='results/grid_results/performance_metrics_extended_raw.csv', help='Path to the extended performance metrics CSV.')
    parser.add_argument('--out_dir', type=str, default='images/detailed', help='Directory to save output plots.')
    parser.add_argument('--fixed_M', type=int, default=500, help='Fixed M_pairs to filter by.')
    parser.add_argument('--fixed_dt', type=float, default=0.05, help='Fixed dt_meas to filter by.')
    args = parser.parse_args()

    sns.set_theme(style="ticks", context='paper')
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"Error: Could not find {args.input}.")
        print("Please ensure you have run 'plot_grid_heatmaps.py' to generate the extended metrics first.")
        return

    df = pd.read_csv(args.input)

    # Standardize the Regime labels
    if 'Regime' in df.columns:
        df['Regime'] = df.apply(label_window, axis=1)

    # Convert numeric columns to proper types
    numeric_cols = ['TP', 'FP', 'FN', 'TN', 'TPR (Recall)', 'FPR', 'Precision', 'F1-Score', 'SHD', 'AUPRC']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Subset the data based on parameters
    df_subset = df[(df['M_pairs'] == args.fixed_M) & (df['dt_meas'] == args.fixed_dt)]
    
    if df_subset.empty:
        print(f"Warning: No data found for M_pairs={args.fixed_M} and dt_meas={args.fixed_dt}")
        return

    # Plot AUPRC: Transient vs Stationary
    fig, ax = plt.subplots(figsize=(7, 9))
    fontsize=21
    
    # By using the raw data, sns.barplot will automatically compute and plot error bars (95% CI).
    # We add capsize for better visibility of the error bars.
    sns.barplot(data=df_subset, x='N', y='AUPRC', hue='Regime', palette='Set2', capsize=0.1, err_kws={'linewidth': 1.5, 'color': 'black'}, ax=ax)
    
    #ax.set_title(f'AUPRC: Transient vs Stationary (Fixed M={args.fixed_M}, dt={args.fixed_dt})')
    ax.set_xlabel('Number of Nodes (N)', fontsize=fontsize)
    ax.set_ylabel('AUPRC', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    
    out_path = os.path.join(args.out_dir, 'transient_vs_stationary_AUPRC_fixed.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot successfully generated and saved to: {out_path}")

if __name__ == "__main__":
    main()
