import json
import numpy as np

with open("results/high_variance_benchmark.json", "r") as f:
    data = json.load(f)

md = "# High Variance Overfitting Constraint Benchmark\n\n"
md += "This document presents the statistical results (Mean ± Std) across 3 independent trials for each true parameter variance setting.\n"
md += "The data scale is fixed at `M_pairs = 500` (High Data regime) for all experiments.\n\n"

for N, scenarios in data.items():
    md += f"## Network Size: N = {N}\n\n"
    md += "| Variance Scenario | TP | FP | FN | F1-Score | SHD | AUROC | AUPRC | MAE | RBO | True Ent | Inf Ent |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for name, trials in scenarios.items():
        metrics = {}
        for k in trials[0].keys():
            vals = [t[k] for t in trials if t[k] is not None and not np.isnan(t[k])]
            if len(vals) > 0:
                metrics[k] = (np.mean(vals), np.std(vals))
            else:
                metrics[k] = (float('nan'), float('nan'))
        
        def format_metric(k, dec=4):
            mean, std = metrics[k]
            if np.isnan(mean): return "NaN"
            return f"{mean:.{dec}f} ± {std:.{dec}f}"
        
        def format_int(k):
            mean, std = metrics[k]
            return f"{mean:.1f} ± {std:.1f}"

        md += f"| **{name}** | {format_int('TP')} | {format_int('FP')} | {format_int('FN')} | {format_metric('F1-Score')} | {format_int('SHD')} | {format_metric('AUROC')} | {format_metric('AUPRC')} | {format_metric('MAE')} | {format_metric('RBO')} | {format_int('True Entropy')} | {format_int('Inferred Entropy')} |\n"
    md += "\n"

with open("/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/high_variance_walkthrough.md", "w") as f:
    f.write(md)

print("Walkthrough generated successfully.")
