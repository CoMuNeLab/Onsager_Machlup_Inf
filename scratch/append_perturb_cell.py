import nbformat
import os

notebook_path = '/home/francesco/Universita/PhD/PROJECTS/InferenceBio/Wilson_Cowan_Inference_Benchmark.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

markdown_cell = nbformat.v4.new_markdown_cell("""
## 4. In Silico Perturbation Analysis

To evaluate the predictive power of our probabilistic ensemble, we perform an *in silico* perturbation experiment. We apply a strong external stimulus to a specific node (e.g., node 0) and calculate the shift in the steady state of the entire network. We compare the prediction made by the single best point-estimate (MAP) against the expected prediction marginalized over the full posterior ensemble.
""")

code_cell = nbformat.v4.new_code_cell("""
from scipy.integrate import odeint
from scipy.stats import spearmanr

def wc_deriv(x, t, W, I_ext, tau_val):
    net_input = W @ x + I_ext
    # Ensure dimensions match
    return -x / tau_val + (1 - x) * expit(net_input)

t_span = np.linspace(0, 100, 500)
x0 = np.full(N, 0.5)

# Unperturbed I
I_ext_unpert = np.full(N, I)

# Perturbed I (Overexpress node 0)
ko_idx = 0
I_ext_pert = I_ext_unpert.copy()
I_ext_pert[ko_idx] += 5.0

# 1. True system response
true_unpert_traj = odeint(wc_deriv, x0, t_span, args=(W_true, I_ext_unpert, tau))
true_unpert = true_unpert_traj[-1]

true_pert_traj = odeint(wc_deriv, true_unpert, t_span, args=(W_true, I_ext_pert, tau))
true_pert = true_pert_traj[-1]
delta_true = true_pert - true_unpert

# 2. Ensemble prediction
N_ens = len(W_ensemble)
delta_inferred = np.zeros((N_ens, N))
stat_inferred_pert = np.zeros((N_ens, N))

for i, W_s in enumerate(W_ensemble):
    # For inference, the tau parameter is absorbed in theta. 
    # But wait, in the notebook, theta_ensemble is also collected. 
    # Let's use the tau from theta_ensemble[i].
    # theta[0] = log(1/tau) -> tau = exp(-theta[0])
    tau_s = np.exp(-theta_ensemble[i][0])
    
    unpert_traj = odeint(wc_deriv, x0, t_span, args=(W_s, I_ext_unpert, tau_s))
    unpert_ss = unpert_traj[-1]
    
    pert_traj = odeint(wc_deriv, unpert_ss, t_span, args=(W_s, I_ext_pert, tau_s))
    pert_ss = pert_traj[-1]
    
    stat_inferred_pert[i] = pert_ss
    delta_inferred[i] = pert_ss - unpert_ss

mean_delta_inf = delta_inferred.mean(axis=0)

# 3. MAP prediction (we use the last sample, which was assigned to W_inferred)
best_idx = -1
best_delta_inf = delta_inferred[best_idx]

# Mask the perturbed node for evaluation
mask = np.ones(N, dtype=bool)
mask[ko_idx] = False

spearman_best, _ = spearmanr(delta_true[mask], best_delta_inf[mask])
spearman_ens, _ = spearmanr(delta_true[mask], mean_delta_inf[mask])

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

min_val = min(delta_true[mask].min(), mean_delta_inf[mask].min(), best_delta_inf[mask].min()) - 0.05
max_val = max(delta_true[mask].max(), mean_delta_inf[mask].max(), best_delta_inf[mask].max()) + 0.05

# Panel A: Scatter Plot
ax = axes[0]
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7, label="Perfect Prediction")

ax.scatter(delta_true[mask], best_delta_inf[mask], color='crimson', s=150, edgecolor='white', 
           label=f"MAP Prediction ($\\rho$ = {spearman_best:.2f})", marker='^', zorder=3)

lower_bound = np.percentile(delta_inferred, 2.5, axis=0)
upper_bound = np.percentile(delta_inferred, 97.5, axis=0)
yerr = [mean_delta_inf - lower_bound, upper_bound - mean_delta_inf]

ax.errorbar(delta_true[mask], mean_delta_inf[mask], yerr=[yerr[0][mask], yerr[1][mask]],
            fmt='o', color='dodgerblue', ecolor='dodgerblue', elinewidth=2, capsize=4,
            markersize=12, markeredgecolor='white', label=f"Ensemble Expected ($\\rho$ = {spearman_ens:.2f})", zorder=2)

ax.set_xlabel(r"True $\Delta$ Steady State")
ax.set_ylabel(r"Predicted $\Delta$ Steady State")
ax.set_title("Systemic Shift Prediction: MAP vs Ensemble")
ax.legend(loc="upper left")

# Panel B: KDE Density for a target node
# Find a node (other than ko_idx) with the largest true shift
target_node = np.argmax(np.abs(delta_true * mask))
ax = axes[1]
sns.kdeplot(stat_inferred_pert[:, target_node], ax=ax, fill=True, color='lightblue', alpha=0.5, lw=2, label="Ensemble KDE")

ax.axvline(true_pert[target_node], color='darkgray', linestyle='-', lw=3, label=f"True State")
ax.axvline(stat_inferred_pert[best_idx, target_node], color='crimson', linestyle='--', lw=2, label="MAP Prediction")
ax.axvline(stat_inferred_pert[:, target_node].mean(), color='dodgerblue', linestyle=':', lw=2, label="Ensemble Mean")

ax.set_xlabel(f"Perturbed Steady State (Node {target_node})")
ax.set_ylabel("Probability Density")
ax.set_title(f"Resolving Uncertainty with Ensembles (Node {target_node})")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
""")

nb.cells.extend([markdown_cell, code_cell])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Cells successfully appended to notebook.")
