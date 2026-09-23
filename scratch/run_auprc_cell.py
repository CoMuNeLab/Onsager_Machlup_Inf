import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style='whitegrid', context='talk')
os.makedirs('images/detailed', exist_ok=True)

df = pd.read_csv('results/grid_results/performance_metrics_extended.csv')

def label_window(row):
    if row['Regime'] == 'Transient':
        return 'T'
    else:
        return 'S'

# The extended csv already has 'Regime' as 'Transient'/'Stationary'
df['Regime'] = df.apply(label_window, axis=1)

df = df[df['M_pairs'] <= 1000]
df = df.reset_index(drop=True)

numeric_cols = ['TP', 'FP', 'FN', 'TN', 'TPR (Recall)', 'FPR', 'Precision', 'F1-Score', 'SHD', 'AUPRC']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

fixed_M = 500
fixed_dt = 0.05
df_subset = df[(df['M_pairs'] == fixed_M) & (df['dt_meas'] == fixed_dt)]

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_subset, x='N', y='AUPRC', hue='Regime', palette='Set2', ax=ax)
ax.set_title(f'AUPRC: Transient vs Stationary (Fixed M={fixed_M}, dt={fixed_dt})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('images/detailed/transient_vs_stationary_AUPRC_fixed.png', dpi=300, bbox_inches='tight')

