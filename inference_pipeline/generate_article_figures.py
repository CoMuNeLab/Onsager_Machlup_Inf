import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set publication quality aesthetics
sns.set_theme(style="white", context="paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

output_dir = '/home/francesco/Universita/PhD/PROJECTS/InferenceBio/images/'
os.makedirs(output_dir, exist_ok=True)

def generate_heatmap_figure():
    # Load the saved matrices
    try:
        data = np.load('/home/francesco/Universita/PhD/PROJECTS/InferenceBio/inference_pipeline/dream4_10node_results.npz')
    except Exception as e:
        print("Could not load npz file. Ensure inference was run.", e)
        return

    W_true = data['W_true']
    edge_probabilities = data['edge_probabilities']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot True Edges
    sns.heatmap(W_true, ax=axes[0], cmap='Greys', cbar=False, linewidths=.5, linecolor='lightgray')
    axes[0].set_title('True Biological Network\n(DREAM4 Gold Standard)', pad=15)
    axes[0].set_xlabel('Source Gene')
    axes[0].set_ylabel('Target Gene')
    axes[0].set_xticks(np.arange(10) + 0.5)
    axes[0].set_xticklabels([f"G{i+1}" for i in range(10)], rotation=90)
    axes[0].set_yticks(np.arange(10) + 0.5)
    axes[0].set_yticklabels([f"G{i+1}" for i in range(10)], rotation=0)
    
    # Plot Inferred Probabilities
    sns.heatmap(edge_probabilities, ax=axes[1], cmap='Blues', vmin=0, vmax=1, linewidths=.5, linecolor='lightgray',
                cbar_kws={'label': 'Posterior Edge Probability'})
    axes[1].set_title('Inferred Ensemble Probabilities\n(MCMC Posterior)', pad=15)
    axes[1].set_xlabel('Source Gene')
    axes[1].set_ylabel('Target Gene')
    axes[1].set_xticks(np.arange(10) + 0.5)
    axes[1].set_xticklabels([f"G{i+1}" for i in range(10)], rotation=90)
    axes[1].set_yticks(np.arange(10) + 0.5)
    axes[1].set_yticklabels([f"G{i+1}" for i in range(10)], rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig1_probabilistic_vs_true_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Generated Fig1_probabilistic_vs_true_heatmap.png")

def generate_timeseries_figure():
    filepath = '/tmp/BSBL-TSK-FS_code/insilico_size10_1_timeseries.tsv'
    try:
        df = pd.read_csv(filepath, sep='\t')
    except Exception as e:
        print("Could not load TSV file.", e)
        return
        
    data = df.values
    timepoints = 21
    
    # Plot the first perturbation response
    traj_1 = data[:timepoints, :]
    
    plt.figure(figsize=(10, 5))
    time_axis = np.arange(timepoints) * 50 # DREAM4 dt is 50
    
    for i in range(10):
        plt.plot(time_axis, traj_1[:, i], marker='o', markersize=4, linewidth=2, label=f'G{i+1}')
        
    plt.title('Gene Expression Dynamics (DREAM4 Size-10, Perturbation 1)', pad=15)
    plt.xlabel('Simulation Time')
    plt.ylabel('Normalized Expression')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig2_dream4_dynamics.png'), bbox_inches='tight')
    plt.close()
    print("Generated Fig2_dream4_dynamics.png")

if __name__ == '__main__':
    generate_heatmap_figure()
    generate_timeseries_figure()
