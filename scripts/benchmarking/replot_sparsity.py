import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def calc_auprc(y_true_binary, y_scores, N):
    mask = ~np.eye(N, dtype=bool)
    y_true_b = y_true_binary[mask]
    y_s = y_scores[mask]
    precision, recall, _ = precision_recall_curve(y_true_b, y_s)
    return auc(recall, precision)

def calc_auroc(y_true_binary, y_scores, N):
    mask = ~np.eye(N, dtype=bool)
    y_true_b = y_true_binary[mask]
    y_s = y_scores[mask]
    fpr, tpr, _ = roc_curve(y_true_b, y_s)
    return auc(fpr, tpr)

def calc_shd(W_true_binary, edge_probabilities, threshold=0.5):
    N = W_true_binary.shape[0]
    mask = ~np.eye(N, dtype=bool)
    true_b = W_true_binary[mask]
    inf_b = (edge_probabilities[mask] >= threshold).astype(int)
    fp = np.sum((true_b == 0) & (inf_b == 1))
    fn = np.sum((true_b == 1) & (inf_b == 0))
    return fp + fn

def calc_mae(W_true, W_inferred, N):
    mask = ~np.eye(N, dtype=bool)
    return np.mean(np.abs(W_true[mask] - W_inferred[mask]))

def calc_spearman(W_true, W_inferred, N):
    mask = ~np.eye(N, dtype=bool)
    return spearmanr(W_true[mask], W_inferred[mask])[0]


def load_trials(results_dir, N, M, dt_meas):
    results = {
        'Global': {'auprc': [], 'auroc': [], 'shd': [], 'spearman': [], 'mae': [], 'trials': []},
        'Local': {'auprc': [], 'auroc': [], 'shd': [], 'spearman': [], 'mae': [], 'trials': []}
    }

    for mode in ['global', 'local']:
        key = 'Global' if mode == 'global' else 'Local'
        pattern = os.path.join(results_dir, f"N_{N}_M_{M}_dt_{dt_meas}_mode_{mode}_trial_*", "results.npz")
        npz_files = sorted(glob.glob(pattern))

        for file_path in npz_files:
            try:
                data = np.load(file_path)
                W_true = data['W_true']
                W_true_b = data['W_true_binary'] if 'W_true_binary' in data else (W_true != 0).astype(int)
                ep = data['edge_probabilities']
                wm = data['weight_mean']
                
                scores = ep * np.abs(wm)
                W_inf = ep * wm

                pr = calc_auprc(W_true_b, scores, N)
                roc = calc_auroc(W_true_b, scores, N)
                shd = calc_shd(W_true_b, ep, threshold=0.5)
                mae = calc_mae(W_true, W_inf, N)
                spearman = calc_spearman(W_true, W_inf, N)

                results[key]['auprc'].append(pr)
                results[key]['auroc'].append(roc)
                results[key]['shd'].append(shd)
                results[key]['mae'].append(mae)
                results[key]['spearman'].append(spearman)
                
                trial_id = int(os.path.basename(os.path.dirname(file_path)).split("_trial_")[-1])
                results[key]['trials'].append(trial_id)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return results


def replot():
    parser = argparse.ArgumentParser(description="Replot Sparsity / Overfitting Benchmark")
    default_results = os.path.join(PROJECT_ROOT, "results", "benchmarking", "regularization", "local_global")
    default_out = os.path.join(PROJECT_ROOT, "images", "benchmarking")
    parser.add_argument("--results_dir", type=str, default=default_results, help="Directory with saved trials")
    parser.add_argument("--N", type=int, default=10, help="Filter: Number of nodes N (default: 10)")
    parser.add_argument("--M", type=int, default=500, help="Filter: Number of samples M (default: 500)")
    parser.add_argument("--dt_meas", type=float, default=0.05, help="Filter: Measurement interval dt_meas (default: 0.05)")
    parser.add_argument("--top_metric", type=str, default="shd", choices=["shd", "spearman", "mae", "auroc"],
                        help="Metric to plot in top panel (default: shd)")
    parser.add_argument("--bottom_metric", type=str, default="auprc", choices=["auprc", "auroc", "spearman", "mae"],
                        help="Metric to plot in bottom panel (default: auprc)")
    parser.add_argument("--out_dir", type=str, default=default_out, help="Directory to save plots")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        # Fallback to results/local_global if needed
        fallback = os.path.join(PROJECT_ROOT, "results", "local_global")
        if os.path.exists(fallback):
            results_dir = fallback

    results = load_trials(results_dir, args.N, args.M, args.dt_meas)

    n_g = len(results['Global']['trials'])
    n_l = len(results['Local']['trials'])

    if n_g == 0 and n_l == 0:
        print(f"No results found in '{args.results_dir}' for N={args.N}, M={args.M}, dt_meas={args.dt_meas}.")
        return

    print(f"\n=======================================================")
    print(f" Loaded {n_g} Global and {n_l} Local trials for N={args.N}, M={args.M}, dt_meas={args.dt_meas}")
    print(f"=======================================================")
    
    for mode in ['Global', 'Local']:
        print(f"--- {mode} Mode (n={len(results[mode]['trials'])}) ---")
        for m in ['shd', 'auprc', 'auroc', 'spearman', 'mae']:
            vals = results[mode][m]
            if len(vals) > 0:
                print(f"  {m.upper():8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Plotting (Publication Quality)
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    fontsize = 22

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))

    metric_labels = {
        'shd': 'SHD (Lower is better)',
        'auprc': 'AUPRC',
        'auroc': 'AUROC',
        'spearman': 'Spearman Correlation',
        'mae': 'MAE'
    }

    # Top plot
    ax_top = axes[0]
    data_top = [results['Global'][args.top_metric], results['Local'][args.top_metric]]
    sns.violinplot(data=data_top, ax=ax_top, palette=['#4C72B0', '#DD8452'], inner=None, linewidth=2)
    sns.stripplot(data=data_top, ax=ax_top, color='black', size=10, jitter=True, alpha=0.7)
    ax_top.set_xticks([0, 1])
    ax_top.set_xticklabels(['Global', 'Local'], rotation=0)
    ax_top.set_ylabel(metric_labels.get(args.top_metric, args.top_metric.upper()), fontsize=fontsize)
    ax_top.tick_params(axis='both', which='major', labelsize=fontsize)
    sns.despine(ax=ax_top)

    # Bottom plot
    ax_bot = axes[1]
    data_bot = [results['Global'][args.bottom_metric], results['Local'][args.bottom_metric]]
    sns.violinplot(data=data_bot, ax=ax_bot, palette=['#4C72B0', '#DD8452'], inner=None, linewidth=2)
    sns.stripplot(data=data_bot, ax=ax_bot, color='black', size=10, jitter=True, alpha=0.7)
    ax_bot.set_xticks([0, 1])
    ax_bot.set_xticklabels(['Global', 'Local'], rotation=0)
    ax_bot.set_ylabel(metric_labels.get(args.bottom_metric, args.bottom_metric.upper()), fontsize=fontsize)
    ax_bot.tick_params(axis='both', which='major', labelsize=fontsize)
    if args.bottom_metric in ['auprc', 'auroc']:
        ax_bot.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax_bot)

    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_path_png = os.path.join(args.out_dir, f'sparsity_comparison_{args.top_metric}_{args.bottom_metric}.png')
    plot_path_pdf = os.path.join(args.out_dir, f'sparsity_comparison_{args.top_metric}_{args.bottom_metric}.pdf')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to:\n - {plot_path_png}\n - {plot_path_pdf}")


if __name__ == "__main__":
    replot()
