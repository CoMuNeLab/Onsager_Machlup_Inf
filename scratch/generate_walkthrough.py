import json

with open("results/complexity_limits_benchmark.json", "r") as f:
    data = json.load(f)

md = "# Complexity Limits Benchmark Results\n\n"
md += "This document presents the full benchmarking results demonstrating the limits of local parameter inference across multiple network scales and scenarios.\n\n"

for N, scenarios in data.items():
    md += f"## Network Size: N = {N}\n\n"
    
    # Table Header
    md += "| Scenario | TP | FP | FN | F1-Score | SHD | AUROC | AUPRC | MAE | RBO | True Ent | Inf Ent |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for name, metrics in scenarios.items():
        auroc = f"{metrics['AUROC']:.4f}" if metrics['AUROC'] == metrics['AUROC'] else "NaN"
        auprc = f"{metrics['AUPRC']:.4f}" if metrics['AUPRC'] == metrics['AUPRC'] else "NaN"
        md += f"| **{name}** | {metrics['TP']} | {metrics['FP']} | {metrics['FN']} | {metrics['F1-Score']:.4f} | {metrics['SHD']} | {auroc} | {auprc} | {metrics['MAE']:.4f} | {metrics['RBO']:.4f} | {metrics['True Entropy']:.0f} | {metrics['Inferred Entropy']:.0f} |\n"
    
    md += "\n"

with open("/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/walkthrough.md", "w") as f:
    f.write(md)

print("Walkthrough generated successfully.")
