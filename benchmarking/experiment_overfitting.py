import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, roc_curve, precision_recall_curve, auc

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
import graph_tool.all as gt
from gene_expr import GeneExprBlockState

def run_experiment():
    np.random.seed(42)
    N = 20
    M_train = 50  # sparse training data to encourage overfitting
    M_test = 50
    dt_meas = 0.5
    dt_sim = 0.005
    dt_sample = 0.5
    
    # 1. Generate Ground Truth Network
    print("Generating network...")
    W_true = generate_scale_free_W(N, m=1, seed=48)
    W_true_binary = (W_true != 0).astype(int)
    
    # 2. Simulate Training and Test Data (Transient Region)
    print("Simulating transient training data...")
    s_train, _, _, _, _ = simulate_realistic_pairs(
        W_true, M=M_train, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
        D_base=0.02, mean_d=0.5, std_d=0.0, mean_V=2.0, std_V=0.0, seed=100, t_window=(0.0, 3.0)
    )
    
    print("Simulating transient test data...")
    s_test, _, _, _, _ = simulate_realistic_pairs(
        W_true, M=M_test, dt_sim=dt_sim, dt_sample=dt_sample, dt_meas=dt_meas, 
        D_base=0.02, mean_d=0.5, std_d=0.0, mean_V=2.0, std_V=0.0, seed=200, t_window=(0.0, 3.0)
    )
    
    # Prepare data arrays for ML models
    X_train_in = np.array([pair[:, 0] for pair in s_train])
    X_train_out = np.array([pair[:, 1] for pair in s_train])
    dX_train = (X_train_out - X_train_in) / dt_meas
    
    X_test_in = np.array([pair[:, 0] for pair in s_test])
    X_test_out = np.array([pair[:, 1] for pair in s_test])
    dX_test = (X_test_out - X_test_in) / dt_meas
    
    results = {}
    
    def calc_metrics(edge_scores):
        mask = ~np.eye(N, dtype=bool)
        y_true = W_true_binary[mask]
        y_scores = edge_scores[mask]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        return fpr, tpr, roc_auc, precision, recall, pr_auc

    # ---------------------------------------------------------
    # Method 1: Random Forest (dynGENIE3-like)
    # ---------------------------------------------------------
    print("Running Random Forest (dynGENIE3 equivalent)...")
    rf_edge_scores = np.zeros((N, N))
    rf_train_preds = np.zeros_like(X_train_out)
    rf_test_preds = np.zeros_like(X_test_out)
    
    for i in range(N):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_in, X_train_out[:, i])
        rf_edge_scores[i, :] = rf.feature_importances_
        rf_train_preds[:, i] = rf.predict(X_train_in)
        rf_test_preds[:, i] = rf.predict(X_test_in)
        
    fpr, tpr, roc_auc, precision, recall, pr_auc = calc_metrics(rf_edge_scores)
    rf_train_mse = mean_squared_error(X_train_out, rf_train_preds)
    rf_test_mse = mean_squared_error(X_test_out, rf_test_preds)
    
    results['dynGENIE3 (RF)'] = {
        'AUPRC': pr_auc, 'AUROC': roc_auc,
        'Train MSE': rf_train_mse, 'Test MSE': rf_test_mse,
        'FPR': fpr, 'TPR': tpr, 'Precision': precision, 'Recall': recall,
        'W': rf_edge_scores
    }
    
    # ---------------------------------------------------------
    # Method 2: Sparse ODE / Lasso
    # ---------------------------------------------------------
    print("Running Sparse ODE (Lasso)...")
    lasso_edge_scores = np.zeros((N, N))
    lasso_train_preds = np.zeros_like(dX_train)
    lasso_test_preds = np.zeros_like(dX_test)
    
    for i in range(N):
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_train_in, dX_train[:, i])
        lasso_edge_scores[i, :] = np.abs(lasso.coef_)
        lasso_train_preds[:, i] = lasso.predict(X_train_in)
        lasso_test_preds[:, i] = lasso.predict(X_test_in)
        
    fpr, tpr, roc_auc, precision, recall, pr_auc = calc_metrics(lasso_edge_scores)
    
    lasso_X_train_preds = X_train_in + lasso_train_preds * dt_meas
    lasso_X_test_preds = X_test_in + lasso_test_preds * dt_meas
    
    lasso_train_mse = mean_squared_error(X_train_out, lasso_X_train_preds)
    lasso_test_mse = mean_squared_error(X_test_out, lasso_X_test_preds)
    
    results['Sparse ODE (Lasso)'] = {
        'AUPRC': pr_auc, 'AUROC': roc_auc,
        'Train MSE': lasso_train_mse, 'Test MSE': lasso_test_mse,
        'FPR': fpr, 'TPR': tpr, 'Precision': precision, 'Recall': recall,
        'W': lasso_edge_scores
    }
    
    # ---------------------------------------------------------
    # Method 3: Graph-Tool SDE
    # ---------------------------------------------------------
    print("Running Graph-Tool SDE...")
    from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
    
    g_init = gt.Graph(directed=True)
    g_init.add_vertex(N)
    
    # Add ~5% density of random edges to jump-start the MCMC
    num_initial_edges = max(1, int(N * N * 0.05))
    possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
    np.random.shuffle(possible_edges)
    for u, v in possible_edges[:num_initial_edges]:
        g_init.add_edge(u, v)
        
    initial_theta = [0.0, 0.0, np.log(0.05)]
    
    state = GeneExprBlockState(s_train, dt=dt_meas, g=g_init, state_args=dict(deg_corr=True), theta=initial_theta, use_em_likelihood=True)
    
    # 50 burn in, 100 sampling
    state = run_burn_in(state, n_burnin=50, theta_mode="global")
    W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, theta_mode="global")
    
    edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
    gt_edge_scores = edge_probabilities
    
    fpr, tpr, roc_auc, precision, recall, pr_auc = calc_metrics(gt_edge_scores)
    
    results['Graph-Tool SDE'] = {
        'AUPRC': pr_auc, 'AUROC': roc_auc,
        'Train MSE': np.nan, 'Test MSE': np.nan,
        'FPR': fpr, 'TPR': tpr, 'Precision': precision, 'Recall': recall,
        'W': gt_edge_scores
    }
    
    print("\n--- Benchmark Results ---")
    for method, metrics in results.items():
        print(f"{method}:")
        print(f"  AUPRC:     {metrics['AUPRC']:.4f}")
        print(f"  AUROC:     {metrics['AUROC']:.4f}")
        if not np.isnan(metrics['Test MSE']):
            print(f"  Train MSE: {metrics['Train MSE']:.4f}")
            print(f"  Test MSE:  {metrics['Test MSE']:.4f}")
            
    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    os.makedirs(os.path.join(os.path.dirname(__file__), 'plots'), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. PR Curve
    ax_pr = axes[0, 0]
    for method, res in results.items():
        ax_pr.plot(res['Recall'], res['Precision'], label=f"{method} (AUC={res['AUPRC']:.2f})")
    ax_pr.set_xlabel('Recall (TPR)')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()
    
    # 2. ROC Curve
    ax_roc = axes[0, 1]
    for method, res in results.items():
        ax_roc.plot(res['FPR'], res['TPR'], label=f"{method} (AUC={res['AUROC']:.2f})")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    
    # 3. Bar plot for Test MSE
    ax_mse = axes[0, 2]
    methods_mse = [m for m in results.keys() if not np.isnan(results[m]['Test MSE'])]
    mses = [results[m]['Test MSE'] for m in methods_mse]
    ax_mse.bar(methods_mse, mses, color=['#ff9999', '#66b3ff'])
    ax_mse.set_ylabel('Test MSE (Data Fitting)')
    ax_mse.set_title('Out-of-sample Prediction Error')
    for i, v in enumerate(mses):
        ax_mse.text(i, v + max(mses)*0.02, f"{v:.4f}", ha='center')
        
    # Matrix Plots
    ax_mat1 = axes[1, 0]
    sns.heatmap(W_true_binary, ax=ax_mat1, cmap='Greys', cbar=False)
    ax_mat1.set_title('True Binary Network')
    
    ax_mat2 = axes[1, 1]
    sns.heatmap(results['dynGENIE3 (RF)']['W'], ax=ax_mat2, cmap='Greens', cbar=False)
    ax_mat2.set_title('Inferred dynGENIE3 (RF)\n(Feature Importances)')
    
    ax_mat3 = axes[1, 2]
    sns.heatmap(results['Graph-Tool SDE']['W'], ax=ax_mat3, cmap='Reds', cbar=False)
    ax_mat3.set_title('Inferred SDE\n(Edge Probabilities)')
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'plots', 'transient_overfitting_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    run_experiment()
