import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, roc_curve, precision_recall_curve, auc
from scipy.stats import spearmanr

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference_pipeline.data_generation.network_generator import generate_scale_free_W
from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
import graph_tool.all as gt
from gene_expr import GeneExprBlockState

def run_experiment(run_inference=True):
    np.random.seed(42)
    N = 10
    M_train = 50  # sparse training data to encourage overfitting
    M_test = 50
    dt_sim = 0.005
    dt_sample = 0.5
    dt_meas = 0.5
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'plots'), exist_ok=True)
    results_path = os.path.join(os.path.dirname(__file__), 'plots', 'overfitting_results.pkl')

    if run_inference:
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
        
        def calc_all_metrics(y_true_binary, y_true_continuous, y_scores, y_inferred_continuous=None):
            mask = ~np.eye(N, dtype=bool)
            y_true_b = y_true_binary[mask]
            y_s = y_scores[mask]
            
            fpr, tpr, _ = roc_curve(y_true_b, y_s)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y_true_b, y_s)
            pr_auc = auc(recall, precision)
            
            spearman_corr, _ = spearmanr(np.abs(y_true_continuous[mask]), np.abs(y_s))
            
            sign_acc = np.nan
            if y_inferred_continuous is not None:
                true_edges_mask = (y_true_b == 1)
                if np.any(true_edges_mask):
                    true_signs = np.sign(y_true_continuous[mask][true_edges_mask])
                    inf_signs = np.sign(y_inferred_continuous[mask][true_edges_mask])
                    sign_acc = np.mean(true_signs == inf_signs)
            
            return fpr, tpr, roc_auc, precision, recall, pr_auc, spearman_corr, sign_acc

        # ---------------------------------------------------------
        # Method 1: Random Forest (dynGENIE3-like)
        # ---------------------------------------------------------
        print("Running Random Forest (dynGENIE3 equivalent)...")
        rf_edge_scores = np.zeros((N, N))
        rf_train_preds = np.zeros_like(X_train_out)
        rf_test_preds = np.zeros_like(X_test_out)
        
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train_in[:, mask], X_train_out[:, i])
            rf_edge_scores[i, mask] = rf.feature_importances_
            rf_train_preds[:, i] = rf.predict(X_train_in[:, mask])
            rf_test_preds[:, i] = rf.predict(X_test_in[:, mask])
            
        fpr, tpr, roc_auc, precision, recall, pr_auc, spearman, sign_acc = calc_all_metrics(
            W_true_binary, W_true, rf_edge_scores, y_inferred_continuous=None
        )
        rf_train_mse = mean_squared_error(X_train_out, rf_train_preds)
        rf_test_mse = mean_squared_error(X_test_out, rf_test_preds)
        
        results['dynGENIE3 (RF)'] = {
            'AUPRC': pr_auc, 'AUROC': roc_auc,
            'Train MSE': rf_train_mse, 'Test MSE': rf_test_mse,
            'Spearman': spearman, 'Sign Accuracy': sign_acc,
            'FPR': fpr, 'TPR': tpr, 'Precision': precision, 'Recall': recall,
            'W': rf_edge_scores
        }
        
        # ---------------------------------------------------------
        # Method 2: Sparse ODE / Lasso
        # ---------------------------------------------------------
        print("Running Sparse ODE (Lasso)...")
        lasso_edge_scores = np.zeros((N, N))
        lasso_coefs_signed = np.zeros((N, N))
        lasso_train_preds = np.zeros_like(dX_train)
        lasso_test_preds = np.zeros_like(dX_test)
        
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            lasso = Lasso(alpha=0.01, random_state=42)
            lasso.fit(X_train_in[:, mask], dX_train[:, i])
            lasso_edge_scores[i, mask] = np.abs(lasso.coef_)
            lasso_coefs_signed[i, mask] = lasso.coef_
            lasso_train_preds[:, i] = lasso.predict(X_train_in[:, mask])
            lasso_test_preds[:, i] = lasso.predict(X_test_in[:, mask])
            
        fpr, tpr, roc_auc, precision, recall, pr_auc, spearman, sign_acc = calc_all_metrics(
            W_true_binary, W_true, lasso_edge_scores, y_inferred_continuous=lasso_coefs_signed
        )
        
        lasso_X_train_preds = X_train_in + lasso_train_preds * dt_meas
        lasso_X_test_preds = X_test_in + lasso_test_preds * dt_meas
        
        lasso_train_mse = mean_squared_error(X_train_out, lasso_X_train_preds)
        lasso_test_mse = mean_squared_error(X_test_out, lasso_X_test_preds)
        
        results['Sparse ODE (Lasso)'] = {
            'AUPRC': pr_auc, 'AUROC': roc_auc,
            'Train MSE': lasso_train_mse, 'Test MSE': lasso_test_mse,
            'Spearman': spearman, 'Sign Accuracy': sign_acc,
            'FPR': fpr, 'TPR': tpr, 'Precision': precision, 'Recall': recall,
            'W': lasso_edge_scores
        }
        
        # ---------------------------------------------------------
        # Method 3: Graph-Tool SDE
        # ---------------------------------------------------------
        print("Running Graph-Tool SDE...")
        from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats
        
        # Instantiate an empty graph in graph-tool format
        g_init = gt.Graph(directed=True)
        g_init.add_vertex(N)

        # Add ~5% density of random edges to jump-start the MCMC and avoid the empty graph trap
        num_initial_edges = max(1, int(N * N * 0.05))
        possible_edges = [(u, v) for u in range(N) for v in range(N) if u != v]
        np.random.shuffle(possible_edges)
        for u, v in possible_edges[:num_initial_edges]:
            g_init.add_edge(u, v)

        # Initialize the custom state with an uninformative prior
        initial_theta = [0.0, 0.0, np.log(0.05)]
        state = GeneExprBlockState(s_train, dt=dt_meas, g=g_init, state_args=dict(deg_corr=False), theta=initial_theta)
        print(f"Initial description length (empty network): {state.entropy():.4f}")

        # Phase 1: Burn-in
        state = run_burn_in(state, n_burnin=50, theta_mode="global")

        # Phase 2: Posterior Sampling
        n_samples = 100
        W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=n_samples, delta_k=1, theta_mode="global")

        # Compute and save ensemble stats
        edge_probabilities, weight_mean, weight_std = compute_ensemble_stats(W_ensemble)
        gt_edge_scores = edge_probabilities * np.abs(weight_mean)
        
        fpr, tpr, roc_auc, precision, recall, pr_auc, spearman, sign_acc = calc_all_metrics(
            W_true_binary, W_true, gt_edge_scores, y_inferred_continuous=weight_mean
        )
        
        # --- Compute GT MSE ---
        def sigmoid_np(x):
            x = np.clip(x, -50, 50)
            return 1.0 / (1.0 + np.exp(-x))
            
        theta_mean = np.mean(theta_ensemble, axis=0)
        d_mean = np.exp(theta_mean[0])
        V_mean = np.exp(theta_mean[1])
        
        def predict_sde(X_in, W, d, V, dt_meas, dt_sim):
            n_steps = int(np.round(dt_meas / dt_sim))
            X_pred = X_in.copy()
            for _ in range(n_steps):
                drift = -d * X_pred + V * sigmoid_np(np.dot(X_pred, W.T))
                X_pred = np.maximum(X_pred + dt_sim * drift, 0.01)
            return X_pred

        gt_train_preds = predict_sde(X_train_in, weight_mean, d_mean, V_mean, dt_meas, dt_sim)
        gt_test_preds = predict_sde(X_test_in, weight_mean, d_mean, V_mean, dt_meas, dt_sim)
        
        gt_train_mse = mean_squared_error(X_train_out, gt_train_preds)
        gt_test_mse = mean_squared_error(X_test_out, gt_test_preds)
        
        results['Graph-Tool SDE'] = {
            'AUPRC': pr_auc, 'AUROC': roc_auc,
            'Train MSE': gt_train_mse, 'Test MSE': gt_test_mse,
            'Spearman': spearman, 'Sign Accuracy': sign_acc,
            'FPR': fpr, 'TPR': tpr, 'Precision': precision, 'Recall': recall,
            'W': gt_edge_scores
        }
        
        # Save results
        print(f"Saving inference results to {results_path}...")
        with open(results_path, 'wb') as f:
            pickle.dump({'results': results, 'W_true_binary': W_true_binary}, f)

    else:
        # Load from file
        if not os.path.exists(results_path):
            print(f"Error: {results_path} does not exist. Run with --run True first.")
            return
            
        print(f"Loading inference results from {results_path}...")
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
            results = data['results']
            W_true_binary = data['W_true_binary']

    print("\n--- Benchmark Results ---")
    for method, metrics in results.items():
        print(f"{method}:")
        print(f"  AUPRC:     {metrics['AUPRC']:.4f}")
        print(f"  AUROC:     {metrics['AUROC']:.4f}")
        print(f"  Spearman:  {metrics.get('Spearman', np.nan):.4f}")
        if not np.isnan(metrics.get('Sign Accuracy', np.nan)):
            print(f"  Sign Acc:  {metrics['Sign Accuracy']:.4f}")
        else:
            print(f"  Sign Acc:  N/A")
        if not np.isnan(metrics['Test MSE']):
            print(f"  Train MSE: {metrics['Train MSE']:.4f}")
            print(f"  Test MSE:  {metrics['Test MSE']:.4f}")
            
    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 3].axis('off')  # Hide top-right empty plot
    
    color_map = {
        'dynGENIE3 (RF)': 'tab:green',
        'Sparse ODE (Lasso)': 'tab:blue',
        'Graph-Tool SDE': 'tab:red'
    }
    
    # 1. PR Curve
    ax_pr = axes[0, 0]
    for method, res in results.items():
        ax_pr.plot(res['Recall'], res['Precision'], color=color_map[method], label=f"{method} (AUC={res['AUPRC']:.2f})")
    ax_pr.set_xlabel('Recall (TPR)')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()
    
    # 2. ROC Curve
    ax_roc = axes[0, 1]
    for method, res in results.items():
        ax_roc.plot(res['FPR'], res['TPR'], color=color_map[method], label=f"{method} (AUC={res['AUROC']:.2f})")
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()
    
    # 3. Bar plot for Test MSE
    ax_mse = axes[0, 2]
    methods_mse = [m for m in results.keys() if not np.isnan(results[m]['Test MSE'])]
    mses = [results[m]['Test MSE'] for m in methods_mse]
    colors_bar = [color_map[m] for m in methods_mse]
    ax_mse.bar(methods_mse, mses, color=colors_bar)
    ax_mse.set_ylabel('Test MSE (Data Fitting)')
    ax_mse.set_title('Out-of-sample Prediction Error')
    for i, v in enumerate(mses):
        ax_mse.text(i, v + max(mses)*0.02, f"{v:.4f}", ha='center')
        
    # Matrix Plots
    ax_mat1 = axes[1, 0]
    W_true_plot = W_true_binary.copy()
    np.fill_diagonal(W_true_plot, 0)
    sns.heatmap(W_true_plot, ax=ax_mat1, cmap='Greys', cbar=False)
    ax_mat1.set_title('True Binary Network')
    
    ax_mat2 = axes[1, 1]
    W_rf_plot = results['dynGENIE3 (RF)']['W'].copy()
    np.fill_diagonal(W_rf_plot, 0)
    sns.heatmap(W_rf_plot, ax=ax_mat2, cmap='Greens', cbar=False)
    ax_mat2.set_title('Inferred dynGENIE3 (RF)\n(Feature Importances)')
    
    ax_mat3 = axes[1, 2]
    W_lasso_plot = results['Sparse ODE (Lasso)']['W'].copy()
    np.fill_diagonal(W_lasso_plot, 0)
    sns.heatmap(W_lasso_plot, ax=ax_mat3, cmap='Blues', cbar=False)
    ax_mat3.set_title('Inferred Sparse ODE (Lasso)\n(Absolute Coefficients)')
    
    ax_mat4 = axes[1, 3]
    W_gt_plot = results['Graph-Tool SDE']['W'].copy()
    np.fill_diagonal(W_gt_plot, 0)
    sns.heatmap(W_gt_plot, ax=ax_mat4, cmap='Reds', cbar=False)
    ax_mat4.set_title('Inferred SDE\n(Edge Probabilities)')
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'plots', 'transient_overfitting_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Overfitting Benchmarking")
    parser.add_argument("--run", type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help="True to run inference, False to load existing results and plot")
    args = parser.parse_args()
    
    run_experiment(run_inference=args.run)
