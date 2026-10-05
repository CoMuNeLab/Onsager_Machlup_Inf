import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def replot():
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
    results_path = os.path.join(plot_dir, 'sparsity_results.pkl')
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'rb') as f:
        results = pickle.load(f)

    # Plotting (Publication Quality)
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    fontsize = 30
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 7))
    
    # 1. Spearman Plot
    ax_spearman = axes[0]
    data_spearman = [results['Global']['spearman'], results['Local']['spearman']]
    sns.violinplot(data=data_spearman, ax=ax_spearman, palette=['#4C72B0', '#DD8452'], inner=None, linewidth=2)
    sns.stripplot(data=data_spearman, ax=ax_spearman, color='black', size=10, jitter=True, alpha=0.7)
    ax_spearman.set_xticks([0, 1])
    ax_spearman.set_xticklabels(['Global', 'Local'], rotation=0)
    ax_spearman.set_ylabel('Spearman Correlation', fontsize=fontsize)
    ax_spearman.tick_params(axis='both', which='major', labelsize=fontsize)
    sns.despine(ax=ax_spearman)
    
    # 2. AUPRC Plot
    ax_pr = axes[1]
    data_pr = [results['Global']['auprc'], results['Local']['auprc']]
    sns.violinplot(data=data_pr, ax=ax_pr, palette=['#4C72B0', '#DD8452'], inner=None, linewidth=2)
    sns.stripplot(data=data_pr, ax=ax_pr, color='black', size=10, jitter=True, alpha=0.7)
    
    ax_pr.set_xticks([0, 1])
    ax_pr.set_xticklabels(['Global', 'Local'], rotation=0)
    ax_pr.set_ylabel('AUPRC', fontsize=fontsize)
    ax_pr.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_pr.set_ylim(0.4, 1.05)
    sns.despine(ax=ax_pr)
    
    plt.tight_layout()
    
    # Save as PNG and PDF for publications
    plot_path_png = os.path.join(plot_dir, 'sparsity_comparison_pub.png')
    plot_path_pdf = os.path.join(plot_dir, 'sparsity_comparison_pub.pdf')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Plots successfully regenerated at:\n - {plot_path_png}\n - {plot_path_pdf}")

if __name__ == "__main__":
    replot()
