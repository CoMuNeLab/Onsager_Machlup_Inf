import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import odeint
import graph_tool.all as gt
from scipy.special import expit
import os
import sys

sys.path.append(os.path.abspath('.'))
from gene_expr import GeneExprBlockState
from inference_pipeline.inference.mcmc_engine import run_burn_in, run_posterior_sampling, compute_ensemble_stats

sns.set_theme(style="white", context="paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

np.random.seed(42)
N = 10
p_edge = 0.15

G_true = nx.erdos_renyi_graph(n=N, p=p_edge, directed=True, seed=42)
W_true = nx.to_numpy_array(G_true)
signs = np.random.choice([-1.0, 1.0], size=(N, N))
true_weights = np.random.uniform(0.5, 1.5, size=(N, N))
W_true = W_true * signs * true_weights

tau = 2.0         
I = -1.5          
sigma = 0.05      
dt_sim = 0.01
steps_per_trial = 200  
num_trials = 30        
subsample = 10         

print(f"Simulating Wilson-Cowan Neural Network (N={N})...")
pairs = []
for trial in range(num_trials):
    x = np.random.uniform(0.1, 0.9, size=N)
    trial_traj = []
    for step in range(steps_per_trial):
        net_input = W_true @ x + I
        activation = expit(net_input)
        dx = (-x / tau + (1 - x) * activation) * dt_sim
        noise = sigma * np.sqrt(dt_sim) * np.random.randn(N)
        x = x + dx + noise
        x = np.clip(x, 0.01, 0.99)
        if step % subsample == 0:
            trial_traj.append(x.copy())
    
    for i in range(len(trial_traj) - 1):
        pairs.append(np.column_stack([trial_traj[i], trial_traj[i+1]]))

dt_infer = dt_sim * subsample
g_init = gt.Graph(directed=True)
g_init.add_vertex(N)
g_init.add_edge(0, 1)

initial_theta = [np.log(1/tau), np.log(1.0), np.log(0.01)]
try:
    gt.seed_rng(42)
except Exception:
    pass

state = GeneExprBlockState(pairs, dt=dt_infer, g=g_init, state_args=dict(deg_corr=False), theta=initial_theta, self_loops=False)
state = run_burn_in(state, n_burnin=100, theta_mode="global", beta=1.0, seed=42)
W_ensemble, theta_ensemble = run_posterior_sampling(state, N, n_samples=100, delta_k=1, theta_mode="global", beta=1.0, seed=42)
W_inferred = W_ensemble[-1]

# Perturbation Analysis
def wc_deriv(x, t, W, I_ext, tau_val):
    net_input = W @ x + I_ext
    return -x / tau_val + (1 - x) * expit(net_input)

t_span = np.linspace(0, 100, 500)
x0 = np.full(N, 0.5)

I_ext_unpert = np.full(N, I)
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
    tau_s = np.exp(-theta_ensemble[i][0])
    
    unpert_traj = odeint(wc_deriv, x0, t_span, args=(W_s, I_ext_unpert, tau_s))
    unpert_ss = unpert_traj[-1]
    
    pert_traj = odeint(wc_deriv, unpert_ss, t_span, args=(W_s, I_ext_pert, tau_s))
    pert_ss = pert_traj[-1]
    
    stat_inferred_pert[i] = pert_ss
    delta_inferred[i] = pert_ss - unpert_ss

mean_delta_inf = delta_inferred.mean(axis=0)

# 3. MAP prediction
best_idx = -1
best_delta_inf = delta_inferred[best_idx]

mask = np.ones(N, dtype=bool)
mask[ko_idx] = False

spearman_best, _ = spearmanr(delta_true[mask], best_delta_inf[mask])
spearman_ens, _ = spearmanr(delta_true[mask], mean_delta_inf[mask])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

min_val = min(delta_true[mask].min(), mean_delta_inf[mask].min(), best_delta_inf[mask].min()) - 0.05
max_val = max(delta_true[mask].max(), mean_delta_inf[mask].max(), best_delta_inf[mask].max()) + 0.05

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
ax.set_title("A. Systemic Shift Prediction: MAP vs Ensemble")
ax.legend(loc="upper left")

target_node = np.argmax(np.abs(delta_true * mask))
ax = axes[1]
sns.kdeplot(stat_inferred_pert[:, target_node], ax=ax, fill=True, color='lightblue', alpha=0.5, lw=2, label="Ensemble KDE")
ax.axvline(true_pert[target_node], color='darkgray', linestyle='-', lw=3, label=f"True State")
ax.axvline(stat_inferred_pert[best_idx, target_node], color='crimson', linestyle='--', lw=2, label="MAP Prediction")
ax.axvline(stat_inferred_pert[:, target_node].mean(), color='dodgerblue', linestyle=':', lw=2, label="Ensemble Mean")

ax.set_xlabel(f"Perturbed Steady State (Node {target_node})")
ax.set_ylabel("Probability Density")
ax.set_title(f"B. Resolving Uncertainty with Ensembles (Node {target_node})")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig('/home/francesco/.gemini/antigravity/brain/7f88c374-5bac-42d3-860d-124cc0776b0a/neural_perturbation.png')
print("Saved plot to neural_perturbation.png")
