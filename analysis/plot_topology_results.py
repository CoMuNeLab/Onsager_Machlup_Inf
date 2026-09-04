#!/usr/bin/env python3
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set aesthetic parameters
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)

_proj_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(_proj_dir, 'results_topology_experiment.json')

with open(json_path, 'r') as f:
    data = json.load(f)

# Flatten data
flat_data = []
for d in data:
    flat_data.append({
        'N': d['N'],
        'Topology': d['Topology'],
        'FL_Level': d['FL_Level'],
        'Scale': d['Scale'],
        'AUROC': d['Metrics']['AUROC'],
        'AUPRC': d['Metrics']['AUPRC'],
        'F1': d['Metrics']['F1'],
        'SHD': d['Metrics']['SHD'],
        'Time': d['Time']
    })
df = pd.DataFrame(flat_data)

fig_dir = os.path.join(_proj_dir, 'images')
os.makedirs(fig_dir, exist_ok=True)

# 1. AUROC by N and Topology
plt.figure()
sns.lineplot(data=df, x='N', y='AUROC', hue='Topology', marker='o', errorbar=('ci', 95))
plt.title('AUROC vs Network Size by Topology')
plt.ylim(0.4, 0.7)
plt.savefig(os.path.join(fig_dir, 'auroc_vs_n_by_topo.png'), bbox_inches='tight')
plt.close()

# 2. AUROC by Scale
plt.figure()
sns.lineplot(data=df, x='N', y='AUROC', hue='Scale', marker='s', errorbar=('ci', 95))
plt.title('AUROC vs Network Size by Parameter Scale')
plt.ylim(0.4, 0.7)
plt.savefig(os.path.join(fig_dir, 'auroc_vs_n_by_scale.png'), bbox_inches='tight')
plt.close()

# 3. AUROC by FL_Level
plt.figure()
sns.lineplot(data=df, x='N', y='AUROC', hue='FL_Level', palette='viridis', marker='D', errorbar=('ci', 95))
plt.title('AUROC vs Network Size by Feedback Loop Density')
plt.ylim(0.4, 0.7)
plt.savefig(os.path.join(fig_dir, 'auroc_vs_n_by_fl.png'), bbox_inches='tight')
plt.close()

# 4. F1 by Topology
plt.figure()
sns.lineplot(data=df, x='N', y='F1', hue='Topology', marker='o', errorbar=('ci', 95))
plt.title('F1 Score vs Network Size by Topology')
plt.ylim(0.0, 0.6)
plt.savefig(os.path.join(fig_dir, 'f1_vs_n_by_topo.png'), bbox_inches='tight')
plt.close()

# 5. Combined AUROC Heatmap for N=10 (to see finer interactions)
plt.figure(figsize=(8, 6))
df_n10 = df[df['N'] == 10]
pivot_n10 = df_n10.pivot_table(index='Topology', columns=['Scale', 'FL_Level'], values='AUROC')
sns.heatmap(pivot_n10, annot=True, cmap='coolwarm', vmin=0.4, vmax=0.7)
plt.title('AUROC for N=10: Topology vs (Scale, FL_Level)')
plt.savefig(os.path.join(fig_dir, 'heatmap_n10.png'), bbox_inches='tight')
plt.close()

print("Plots generated successfully in images/")
