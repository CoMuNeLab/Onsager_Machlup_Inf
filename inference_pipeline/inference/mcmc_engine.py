import numpy as np
import scipy.optimize as opt
from gene_expr import GeneExprBlockState

def optimize_global_theta(state):
    S_cur = state.entropy()
    theta_cur = state.theta.get_2d_array([0, 1, 2])[:, 0]
    
    def objective(theta_val):
        arr = state.theta.get_2d_array([0, 1, 2])
        arr[0, :] = theta_val[0]
        arr[1, :] = theta_val[1]
        arr[2, :] = theta_val[2]
        state.theta.set_2d_array(arr)
        return state.entropy()
    
    res = opt.minimize(objective, theta_cur, method='Nelder-Mead', options={'maxiter': 30})
    
    if res.fun < S_cur:
        arr = state.theta.get_2d_array([0, 1, 2])
        arr[0, :] = res.x[0]
        arr[1, :] = res.x[1]
        arr[2, :] = res.x[2]
        state.theta.set_2d_array(arr)
        return S_cur - res.fun
    else:
        arr = state.theta.get_2d_array([0, 1, 2])
        arr[0, :] = theta_cur[0]
        arr[1, :] = theta_cur[1]
        arr[2, :] = theta_cur[2]
        state.theta.set_2d_array(arr)
        return 0.0

def run_burn_in(state, n_burnin=50, optimize_params=True):
    print(f"\n--- PHASE 1: Equilibration ({n_burnin} sweeps) ---")
    for sweep in range(1, n_burnin + 1):
        dS_topo, nmoves = state.mcmc_sweep(niter=10, theta=0)
        dS_param = optimize_global_theta(state) if optimize_params else 0.0
        if sweep % 10 == 0 or sweep == 1:
            print(f"Burn-in Sweep {sweep:2d}/{n_burnin}: Entropy = {state.entropy():8.4f} (dS_topo = {dS_topo:8.4f}, dS_param = {-dS_param:8.4f}, moves = {nmoves:4d})")
    return state

def run_posterior_sampling(state, N, n_samples=100, delta_k=1, optimize_params=True):
    print(f"\n--- PHASE 2: Posterior Sampling ({n_samples} samples, every {delta_k} sweeps) ---")
    W_ensemble = []
    theta_ensemble = []
    
    for sample_idx in range(1, n_samples + 1):
        for _ in range(delta_k):
            dS_topo, nmoves = state.mcmc_sweep(niter=10, theta=0)
            dS_param = optimize_global_theta(state) if optimize_params else 0.0
            
        S_cur = state.entropy()
        
        # Extract the adjacency matrix (weights)
        g_curr = state.get_graph()
        W_curr = np.zeros((N, N))
        x_prop = state.x
        for e in g_curr.edges():
            W_curr[int(e.target()), int(e.source())] = x_prop[e]
        W_ensemble.append(W_curr)
        
        # Extract global parameters
        theta_curr = state.theta.get_2d_array([0, 1, 2])[:, 0]
        theta_ensemble.append(theta_curr)
        
        if sample_idx % 10 == 0 or sample_idx == 1:
            print(f"Sample {sample_idx:3d}/{n_samples}: Entropy = {S_cur:8.4f}")
            
    return np.array(W_ensemble), np.array(theta_ensemble)

def compute_ensemble_stats(W_ensemble):
    edge_exists = (W_ensemble != 0).astype(float)
    edge_probabilities = np.mean(edge_exists, axis=0)
    
    masked_W = np.ma.masked_equal(W_ensemble, 0)
    weight_mean = masked_W.mean(axis=0).filled(0.0)
    weight_std = masked_W.std(axis=0).filled(0.0)
    
    return edge_probabilities, weight_mean, weight_std
