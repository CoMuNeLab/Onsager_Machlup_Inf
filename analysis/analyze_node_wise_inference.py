#!/usr/bin/env python3
"""
Analyze Node-Wise Inference Results.
Produces scatter plots, correlation analyses, and an ANOVA variance decomposition
to evaluate how node-wise trajectory properties (Var, DriftSq, eSNR) predict
the accuracy of inferring incoming edges.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import spearmanr

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

_proj_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(_proj_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

def load_data():
    csv_path = os.path.join(_proj_dir, "results_node_wise_inference.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop rows where AUROC is NaN (nodes with in-degree 0 or N-1)
    df_clean = df.dropna(subset=["AUROC", "AUPRC"]).copy()
    print(f"Loaded dataset: {len(df)} total nodes. {len(df_clean)} nodes after dropping NaNs.")
    return df_clean

def plot_scatter_metrics(df):
    """
    Scatter plots of node-wise AUROC against trajectory metrics.
    Uses hexbin or scatter to handle overlapping points.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Var(x)
    sns.scatterplot(data=df, x="var_x", y="AUROC", hue="regime", style="topology", 
                    alpha=0.6, s=30, ax=axes[0])
    axes[0].set_title("A. Node-Wise AUROC vs. Node Trajectory Variance")
    axes[0].set_xlabel("Node Expression Variance Var(x_i)")
    axes[0].set_ylabel("Node AUROC (Incoming Edges)")
    
    # Mean Squared Drift
    sns.scatterplot(data=df, x="mean_drift_sq", y="AUROC", hue="regime", style="topology",
                    alpha=0.6, s=30, ax=axes[1])
    axes[1].set_title("B. Node-Wise AUROC vs. Mean Squared Drift")
    axes[1].set_xlabel("Mean Squared Drift <|f_i(x)|^2>")
    axes[1].set_ylabel("Node AUROC")
    
    # Empirical SNR
    sns.scatterplot(data=df, x="esnr_vel", y="AUROC", hue="regime", style="topology",
                    alpha=0.6, s=30, ax=axes[2])
    axes[2].set_xscale("log")
    axes[2].set_title("C. Node-Wise AUROC vs. Empirical Velocity SNR")
    axes[2].set_xlabel("Empirical SNR (log scale)")
    axes[2].set_ylabel("Node AUROC")
    
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "fig_node01_scatter_auroc_vs_physics.png")
    plt.savefig(out_path)
    plt.close()
    print(f" -> Saved {out_path}")

def perform_correlation_analysis(df):
    metrics_x = ["var_x", "mean_drift_sq", "esnr_vel", "tdmi", "in_degree"]
    metrics_y = ["AUROC", "AUPRC", "MSE", "F1", "SHD"]
    
    corr_results = []
    
    for y in metrics_y:
        row = {"Target": y}
        for x in metrics_x:
            r, p = spearmanr(df[x], df[y])
            row[x] = r
        corr_results.append(row)
        
    df_corr = pd.DataFrame(corr_results).set_index("Target")
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", center=0, fmt=".3f")
    plt.title("Spearman Rank Correlation: Trajectory Content vs. Inference Performance")
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "fig_node02_correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f" -> Saved {out_path}")
    
    return df_corr

def perform_regression_and_report(df, df_corr):
    """
    Run OLS and write markdown report.
    """
    # Standardize to get standardized beta coefficients for comparison
    df_reg = df.copy()
    for col in ["var_x", "mean_drift_sq", "esnr_vel", "tdmi", "AUROC"]:
        df_reg[col] = (df_reg[col] - df_reg[col].mean()) / df_reg[col].std()
        
    df_reg["log_M"] = np.log10(df_reg["M"])
    
    # Model AUROC using standardized physical parameters
    model = ols("AUROC ~ log_M + var_x + mean_drift_sq + esnr_vel", data=df_reg).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    anova["pct_variance"] = (anova["sum_sq"] / anova["sum_sq"].sum()) * 100.0
    
    # Generate Report
    report_path = os.path.join(_proj_dir, "report_node_wise_inference.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Node-Wise Trajectory Information and Network Inference Report\n\n")
        f.write("This report evaluates whether the possibility of reconstructing the incoming parameters ")
        f.write("associated with a *single node* is predicted by the information content of its *own trajectory*.\n\n")
        
        f.write("## 1. Spearman Rank Correlations\n\n")
        f.write("Correlation between node-specific trajectory metrics and inference performance metrics for that node's incoming edges:\n\n")
        f.write(df_corr.to_string())
        f.write("\n\n")
        
        f.write("## 2. Standardized OLS Regression (AUROC)\n\n")
        f.write("Model: `AUROC ~ log_M + var_x + mean_drift_sq + esnr_vel` (all standardized)\n\n")
        f.write("```text\n")
        f.write(model.summary().as_text())
        f.write("\n```\n\n")
        
        f.write("## 3. ANOVA Variance Decomposition\n\n")
        f.write("Which factor explains the most variance in node-wise AUROC?\n\n")
        f.write("| Factor | Sum of Squares | F | p-value | % Variance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for index, row in anova.iterrows():
            f.write(f"| `{index}` | {row['sum_sq']:.4f} | {row.get('F', np.nan):.2f} | {row.get('PR(>F)', np.nan):.2e} | **{row['pct_variance']:.2f}%** |\n")
            
        f.write("\n## 4. Conclusion\n\n")
        f.write("The analysis clearly shows whether individual node trajectory content, independent of the rest of the network, determines parameter reconstructability.\n")
        
    print(f" -> Saved {report_path}")

def main():
    print("Starting Node-Wise Analysis...")
    df = load_data()
    plot_scatter_metrics(df)
    df_corr = perform_correlation_analysis(df)
    perform_regression_and_report(df, df_corr)
    print("\nSUCCESS: Analysis completed!")

if __name__ == "__main__":
    main()
