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
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # 1. MAE Plot
    ax_mae = axes[0]
    data_mae = [results['Global']['mae'], results['Local']['mae']]
    sns.boxplot(data=data_mae, ax=ax_mae, palette=['#4C72B0', '#DD8452'], showfliers=False, width=0.5, boxprops={'alpha': 0.6}, linewidth=3, medianprops={'color': 'black', 'linewidth': 2})
    sns.stripplot(data=data_mae, ax=ax_mae, color='black', size=12, jitter=True, alpha=0.8)
    ax_mae.set_xticks([0, 1])
    ax_mae.set_xticklabels(['Global', 'Local'], rotation=0)
    ax_mae.set_ylabel('MAE', fontsize=fontsize)
    ax_mae.tick_params(axis='both', which='major', labelsize=fontsize)
    #ax_mae.set_title('Inference Error (Weights)', pad=15)
    sns.despine(ax=ax_mae)
    
    # 2. AUPRC Plot
    ax_pr = axes[1]
    data_pr = [results['Global']['auprc'], results['Local']['auprc']]
    
    sns.boxplot(data=data_pr, ax=ax_pr, palette=['#4C72B0', '#DD8452'], showfliers=False, width=0.5, boxprops={'alpha': 0.6}, linewidth=3, medianprops={'color': 'black', 'linewidth': 2})
    sns.stripplot(data=data_pr, ax=ax_pr, color='black', size=12, jitter=True, alpha=0.8)
    
    ax_pr.set_xticks([0, 1])
    ax_pr.set_xticklabels(['Global', 'Local'], rotation=0)
    ax_pr.set_ylabel('AUPRC', fontsize=fontsize)
    ax_pr.tick_params(axis='both', which='major', labelsize=fontsize)
    ax_pr.set_ylim(0.5, 1.05)
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
