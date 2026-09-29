import json
import numpy as np
import matplotlib.pyplot as plt
import os

with open("results/high_variance_benchmark.json", "r") as f:
    data = json.load(f)

stds = [0.1, 0.5, 1.0, 1.5, 2.0]
scenario_names = [f"std_{std:.1f}" for std in stds]
networks = ["10", "20"]

out_dir = "/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/images"
os.makedirs(out_dir, exist_ok=True)

# Colors
colors = {'10': 'blue', '20': 'orange'}

# 1. F1-Score vs Variance
plt.figure(figsize=(8, 6))
for N in networks:
    means = []
    errs = []
    for sc in scenario_names:
        trials = data[N][sc]
        vals = [t['F1-Score'] for t in trials if t['F1-Score'] is not None and not np.isnan(t['F1-Score'])]
        means.append(np.mean(vals))
        errs.append(np.std(vals))
    plt.errorbar(stds, means, yerr=errs, label=f"N={N}", fmt='-o', capsize=5, color=colors[N], linewidth=2)

plt.xlabel("Heterogeneity (Std. Dev.)", fontsize=14)
plt.ylabel("F1-Score", fontsize=14)
plt.title("Performance vs True Parameter Heterogeneity", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/f1_vs_variance.png", dpi=300)
plt.close()

# 2. False Positives vs Variance
plt.figure(figsize=(8, 6))
for N in networks:
    means = []
    errs = []
    for sc in scenario_names:
        trials = data[N][sc]
        vals = [t['FP'] for t in trials if t['FP'] is not None]
        means.append(np.mean(vals))
        errs.append(np.std(vals))
    plt.errorbar(stds, means, yerr=errs, label=f"N={N}", fmt='-o', capsize=5, color=colors[N], linewidth=2)

plt.xlabel("Heterogeneity (Std. Dev.)", fontsize=14)
plt.ylabel("False Positives", fontsize=14)
plt.title("False Positives vs True Parameter Heterogeneity", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/fp_vs_variance.png", dpi=300)
plt.close()

# 3. Entropy Gap vs Variance
plt.figure(figsize=(8, 6))
for N in networks:
    means = []
    errs = []
    for sc in scenario_names:
        trials = data[N][sc]
        vals = [t['True Entropy'] - t['Inferred Entropy'] for t in trials if t['True Entropy'] is not None]
        means.append(np.mean(vals))
        errs.append(np.std(vals))
    plt.errorbar(stds, means, yerr=errs, label=f"N={N}", fmt='-o', capsize=5, color=colors[N], linewidth=2)

plt.xlabel("Heterogeneity (Std. Dev.)", fontsize=14)
plt.ylabel("Entropy Gap (True - Inferred)", fontsize=14)
plt.title("Overfitting Severity vs Heterogeneity", fontsize=16)
plt.yscale("log")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/entropy_gap_vs_variance.png", dpi=300)
plt.close()

print("Plots generated successfully.")
