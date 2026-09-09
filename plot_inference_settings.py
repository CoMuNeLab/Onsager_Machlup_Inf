import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot results from inference settings experiment")
    parser.add_argument("suffix", type=str, nargs="?", default="", help="Suffix identifying the experiment (e.g. N20_M500_dt05)")
    return parser.parse_args()

def plot_experiment():
    args = parse_args()
    
    suffix = args.suffix
    
    suffix_str = f"_{suffix}" if suffix else ""
    results_file = f"results/model_exp/results{suffix_str}.npz"
    metrics_file = f"results/model_exp/metrics{suffix_str}.txt"
        
    print(f"Loading results from {results_file}...")
    data = np.load(results_file)
    W_true = data['W_true']
    W_true_binary = data['W_true_binary']
    res_1_prob = data['res_1_prob']
    res_1_weight_last = data['res_1_weight_last']
    res_2_prob = data['res_2_prob']
    res_2_weight_last = data['res_2_weight_last']
    res_3_prob = data['res_3_prob']
    res_3_weight_last = data['res_3_weight_last']
    
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            print(f.read())
    else:
        print(f"Metrics file {metrics_file} not found.")
        
    print("Generating visualizations...")
    
    # 1. Binary / Probabilistic Matrices
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    sns.heatmap(W_true_binary, ax=axes[0], cmap="Blues", cbar=False, square=True)
    axes[0].set_title("True Binary Network")
    
    sns.heatmap(res_1_prob, ax=axes[1], cmap="Blues", vmin=0, vmax=1, square=True)
    axes[1].set_title("1. Analytical Global\n(Edge Probabilities)")
    
    sns.heatmap(res_2_prob, ax=axes[2], cmap="Blues", vmin=0, vmax=1, square=True)
    axes[2].set_title("2. E-M Global\n(Edge Probabilities)")
    
    sns.heatmap(res_3_prob, ax=axes[3], cmap="Blues", vmin=0, vmax=1, square=True)
    axes[3].set_title("3. Analytical Local\n(Edge Probabilities)")
    
    plt.tight_layout()
    plt.savefig(f"results/model_exp/binary_matrices_comparison{suffix_str}.png", dpi=300)
    plt.close()
    
    # 2. Weighted Matrices (Using last sample)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    vmax = max(np.max(W_true), np.max(res_1_weight_last), np.max(res_2_weight_last), np.max(res_3_weight_last))
    vmin = min(np.min(W_true), np.min(res_1_weight_last), np.min(res_2_weight_last), np.min(res_3_weight_last))
    vmax_abs = max(abs(vmax), abs(vmin))
    vmax = vmax_abs
    vmin = -vmax_abs
    
    sns.heatmap(W_true, ax=axes[0], cmap="coolwarm", vmin=vmin, vmax=vmax, center=0, cbar=False, square=True)
    axes[0].set_title("True Weighted Network")
    
    sns.heatmap(res_1_weight_last, ax=axes[1], cmap="coolwarm", vmin=vmin, vmax=vmax, center=0, square=True)
    axes[1].set_title("1. Analytical Global\n(Last Weight Sample)")
    
    sns.heatmap(res_2_weight_last, ax=axes[2], cmap="coolwarm", vmin=vmin, vmax=vmax, center=0, square=True)
    axes[2].set_title("2. E-M Global\n(Last Weight Sample)")
    
    sns.heatmap(res_3_weight_last, ax=axes[3], cmap="coolwarm", vmin=vmin, vmax=vmax, center=0, square=True)
    axes[3].set_title("3. Analytical Local\n(Last Weight Sample)")
    
    plt.tight_layout()
    plt.savefig(f"results/model_exp/weighted_matrices_comparison{suffix_str}.png", dpi=300)
    plt.close()
    
    print(f"Plotting complete! Images saved in results/model_exp/")

if __name__ == "__main__":
    plot_experiment()
