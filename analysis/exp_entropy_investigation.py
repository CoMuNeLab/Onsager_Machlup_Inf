import os
import sys
import numpy as np

if os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))

import graph_tool.all as gt
from gene_expr import GeneExprBlockState

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def generate_scale_free_W(N, m=2, seed=42):
    np.random.seed(seed)
    W = np.zeros((N, N))
    out_degrees = np.zeros(N)
    core_size = m + 1
    for i in range(core_size):
        for j in range(core_size):
            if i != j and np.random.rand() > 0.5:
                sign = np.random.choice([-1, 1])
                mag = np.random.uniform(1.0, 3.0)
                W[i, j] = sign * mag
                out_degrees[j] += 1
    for i in range(core_size, N):
        weights = out_degrees[:i] + 1.0
        probs = weights / np.sum(weights)
        regulators = np.random.choice(i, size=m, replace=False, p=probs)
        for reg in regulators:
            sign = np.random.choice([-1, 1])
            mag = np.random.uniform(1.0, 3.0)
            W[i, reg] = sign * mag
            out_degrees[reg] += 1
    return W

def simulate_realistic_trajectories(W, T_total, dt, dt_meas, D_base, mean_d, std_d, mean_V, std_V, seed=123):
    N = W.shape[0]
    np.random.seed(seed)
    d = np.random.lognormal(mean=np.log(mean_d), sigma=std_d, size=N)
    V = np.random.lognormal(mean=np.log(mean_V), sigma=std_V, size=N)
    n_steps = int(T_total / dt)
    x = np.random.uniform(0.1, 1.0, size=N)
    x_history = []
    t_history = []
    
    meas_step = max(1, int(dt_meas / dt))
    
    for step in range(n_steps):
        t = step * dt
        drift = -d * x + V * sigmoid(np.dot(W, x))
        # Multiplicative CIR noise matching the C++ extension exactly
        noise = np.sqrt(D_base * np.maximum(x, 0.01)) * np.random.normal(0, np.sqrt(dt), size=N)
        
        x = x + drift * dt + noise
        x = np.maximum(x, 0.01)
        
        if step % meas_step == 0:
            x_history.append(x.copy())
            t_history.append(t)
            
    return np.array(t_history), np.array(x_history), d, V

def build_g_true(W_true, N):
    g_true = gt.Graph(directed=True)
    g_true.add_vertex(N)
    ep_x = g_true.new_edge_property("double")
    for i in range(N):
        for j in range(N):
            if W_true[i, j] != 0:
                e = g_true.add_edge(j, i)
                ep_x[e] = W_true[i, j]
    g_true.ep["x"] = ep_x
    return g_true

def run_experiment(name, s, dt_eval, g_true, theta_true, fix_theta=False, n_sweeps=30):
    print(f"\n{'='*50}\nEXPERIMENT: {name}\n{'='*50}")
    
    # Evaluate True Structure + True Parameters
    state_true = GeneExprBlockState(s, dt=dt_eval, g=g_true, theta=theta_true)
    S_true = state_true.entropy()
    
    # Evaluate Empty Structure + Default Parameters
    kwargs = {}
    if fix_theta:
        # If fixing theta, we initialize the empty graph with theta_true and freeze it
        kwargs['theta'] = theta_true
        kwargs['tdelta'] = [0.0, 0.0, 0.0]
        kwargs['tdelta_min'] = [0.0, 0.0, 0.0]
        
    state_opt = GeneExprBlockState(s, dt=dt_eval, **kwargs)
    S_init = state_opt.entropy()
    
    best_S = S_init
    best_graph = state_opt.get_graph().copy()
    best_theta = state_opt.theta.copy()
    
    for sweep in range(1, n_sweeps + 1):
        dS, nmoves = state_opt.mcmc_sweep(niter=10)
        S_cur = state_opt.entropy()
        if S_cur < best_S:
            best_S = S_cur
            best_graph = state_opt.get_graph().copy()
            best_theta = state_opt.theta.copy()
            
    # Count edges
    true_edges = g_true.num_edges()
    inferred_edges = best_graph.num_edges()
    
    # Output results
    print(f"S_true (True Graph & True Theta):  {S_true:.4f}")
    print(f"S_init (Empty Graph):              {S_init:.4f}")
    print(f"S_best (Inferred Graph):           {best_S:.4f}")
    print(f"Delta S (S_true - S_best):         {S_true - best_S:.4f}")
    print(f"")
    print(f"True Edges: {true_edges}, Inferred Edges: {inferred_edges}")
    print(f"True Theta:     {list(theta_true)}")
    print(f"Inferred Theta: {list(best_theta)}")
    if fix_theta:
        print("Note: Theta was frozen during optimization.")
    
def main():
    N = 10
    T_total = 50.0
    dt = 0.01
    
    mean_decay = 0.5
    mean_V = 2.0
    D_base = 0.1
    
    # Ensure fully homogeneous simulation for baseline tests
    std_decay = 0.0
    std_V = 0.0
    
    W_true = generate_scale_free_W(N, m=2, seed=42)
    g_true = build_g_true(W_true, N)
    theta_true = [np.log(mean_decay), np.log(mean_V), np.log(D_base)]
    
    # 1. Coarse Sampling Data
    dt_meas_coarse = 0.5
    _, X_coarse, _, _ = simulate_realistic_trajectories(W_true, T_total, dt, dt_meas_coarse, D_base, mean_decay, std_decay, mean_V, std_V)
    s_coarse = X_coarse.T
    
    # 2. Fine Sampling Data
    dt_meas_fine = 0.01
    _, X_fine, _, _ = simulate_realistic_trajectories(W_true, T_total, dt, dt_meas_fine, D_base, mean_decay, std_decay, mean_V, std_V)
    s_fine = X_fine.T

    # Experiment 1, 2, 3: Standard Inference on Coarse Data
    run_experiment("Standard Inference (Coarse dt_meas=0.5)", s_coarse, dt_meas_coarse, g_true, theta_true, fix_theta=False)
    
    # Experiment 5: User Requested - Fix Theta and update only Network on Coarse Data
    run_experiment("Fixed Theta Inference (Update Network Only, Coarse dt_meas=0.5)", s_coarse, dt_meas_coarse, g_true, theta_true, fix_theta=True)

    # Experiment 4: Standard Inference on Fine Data (No Discretization Error)
    # Using fewer steps to save time because N*M is much larger (5000 points)
    # We will just evaluate S_true vs S_init and run a quick MCMC (n_sweeps=5)
    run_experiment("Fine-grained Inference (No Discretization Error, dt_meas=0.01)", s_fine, dt_meas_fine, g_true, theta_true, fix_theta=False, n_sweeps=10)

if __name__ == "__main__":
    main()
