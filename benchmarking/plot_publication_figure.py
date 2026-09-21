import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication quality styling
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

def plot_publication_figure(data_path="data/publication_likelihood_benchmark.csv", out_path="publication_figure.pdf"):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # We have 3 regimes: 'Baseline', 'Stiff', 'High_Noise'
    regimes = ['Baseline', 'Stiff', 'High_Noise']
    regime_titles = {
        'Baseline': r'Baseline ($\gamma=0.5, D=0.02$)',
        'Stiff': r'Stiff Dynamics ($\gamma=2.0, D=0.02$)',
        'High_Noise': r'High Noise ($\gamma=0.5, D=0.2$)'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    
    # Set custom colors and linestyles
    palette = {'Standard': '#1f77b4', 'EM': '#d62728'}  # Blue for Exact, Red for EM
    dashes = {'Standard': '', 'EM': (2, 2)} # Solid for Exact, Dashed for EM
    
    metrics = [('AUROC', axes[0, :]), ('AUPRC', axes[1, :])]
    
    for row_idx, (metric_name, ax_row) in enumerate(metrics):
        for col_idx, regime in enumerate(regimes):
            ax = ax_row[col_idx]
            subset = df[df['Regime'] == regime]
            
            # Plot using seaborn
            sns.lineplot(
                data=subset, x='dt', y=metric_name, hue='Method', style='Method',
                palette=palette, dashes=dashes, errorbar='sd', ax=ax,
                linewidth=2.5, markers=True, markersize=8
            )
            
            # Formatting
            if row_idx == 0:
                ax.set_title(regime_titles[regime], fontsize=16, pad=15)
                ax.set_ylim(0.4, 1.05)
            else:
                ax.set_ylim(0.0, 1.05)
                ax.set_xlabel(r'Measurement Interval $\Delta t$', fontsize=14)
                
            if col_idx == 0:
                ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel('')
                
            # Grid and ticks
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Clean up legends
            if row_idx == 0 and col_idx == 2:
                # Add legend only to top right
                handles, labels = ax.get_legend_handles_labels()
                # Rename 'Standard' to 'Exact CIR' in legend if desired
                labels = ['Exact Transition' if l == 'Standard' else 'Euler-Maruyama' for l in labels]
                ax.legend(handles, labels, title='Likelihood', loc='lower left', bbox_to_anchor=(1.05, 0.5), fontsize=12)
            else:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(right=0.85) # Make room for legend
    
    # Save high-res
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved publication figure to {out_path} and {out_path.replace('.pdf', '.png')}")

if __name__ == "__main__":
    plot_publication_figure()
