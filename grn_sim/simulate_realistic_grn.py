import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Clip to avoid overflow
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def generate_scale_free_W(N, m=2, seed=42):
    """Generates a scale-free directed GRN using directed preferential attachment."""
    np.random.seed(seed)
    W = np.zeros((N, N))
    
    out_degrees = np.zeros(N)
    
    # Initialize a small core network (first m+1 nodes)
    # Give them some random directed edges so they have non-zero out-degrees
    core_size = m + 1
    for i in range(core_size):
        for j in range(core_size):
            if i != j and np.random.rand() > 0.5:
                # Edge from j to i
                sign = np.random.choice([-1, 1])
                mag = np.random.uniform(1.0, 3.0)
                W[i, j] = sign * mag
                out_degrees[j] += 1
                
    # Preferential attachment for the remaining nodes
    for i in range(core_size, N):
        # Probability of choosing an existing node j as a regulator 
        # is proportional to its current out-degree + 1 (to allow nodes with 0 out-degree to be chosen)
        weights = out_degrees[:i] + 1.0
        probs = weights / np.sum(weights)
        
        # Node i chooses m regulators
        regulators = np.random.choice(i, size=m, replace=False, p=probs)
        
        for reg in regulators:
            sign = np.random.choice([-1, 1])
            mag = np.random.uniform(1.0, 3.0)
            W[i, reg] = sign * mag  # reg regulates i
            out_degrees[reg] += 1
            
    return W

def simulate_realistic_trajectories(W, T_total, dt, dt_meas, perturb_time):
    N = W.shape[0]
    
    # 1. Heterogeneous parameters (Log-Normal)
    np.random.seed(123)
    d = np.random.lognormal(mean=np.log(0.5), sigma=0.5, size=N)
    V = np.random.lognormal(mean=np.log(2.0), sigma=0.5, size=N)
    
    D_base = 0.1 # Base noise level
    
    n_steps = int(T_total / dt)
    n_meas = int(T_total / dt_meas)
    
    # Initial condition
    x = np.random.uniform(0.1, 1.0, size=N)
    x_history = []
    t_history = []
    
    # Find the "master regulator" (hub node with highest out-degree)
    # W[i, j] means j regulates i. Thus out-degree of j is the sum over row elements i in column j.
    out_degrees = (W != 0).sum(axis=0)
    hub_node = np.argmax(out_degrees)
    
    meas_step = int(dt_meas / dt)
    
    for step in range(n_steps):
        t = step * dt
        
        # 3. Compute drift
        # The drift includes the exponential decay (-d * x) and saturating production
        drift = -d * x + V * sigmoid(W @ x)
        
        # 4. Multiplicative Noise (Cox-Ingersoll-Ross style)
        # Noise variance scales with the expression level (like transcriptional bursting)
        noise = np.sqrt(2 * D_base * np.maximum(x, 0.01)) * np.random.normal(0, np.sqrt(dt), size=N)
        
        x = x + drift * dt + noise
        x = np.maximum(x, 0.0) # mRNA cannot be negative
        
        if step % meas_step == 0:
            x_history.append(x.copy())
            t_history.append(t)
            
    return np.array(t_history), np.array(x_history), hub_node

def main():
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    
    N = 25  # Size of the network
    W = generate_scale_free_W(N, m=2, seed=42)
    
    T_total = 20.0
    dt = 0.001
    dt_meas = 0.05
    perturb_time = 10.0
    
    print(f"Simulating biologically realistic trajectories for {N} genes...")
    t, X, hub_node = simulate_realistic_trajectories(W, T_total, dt, dt_meas, perturb_time)
    
    # -- Plotting --
    plt.figure(figsize=(14, 7))
    
    # Plot all non-hub genes
    for i in range(N):
        if i != hub_node:
            plt.plot(t, X[:, i], alpha=0.3, linewidth=1.5)
            
    # Highlight the hub gene
    plt.plot(t, X[:, hub_node], color='red', linewidth=3.5, label=f'Master Regulator (Node {hub_node})')
    
    plt.title("Biologically Realistic GRN Trajectories\n(Scale-Free Topology + Heterogeneous Kinetics + Multiplicative Noise)", fontsize=16, pad=15)
    plt.xlabel("Time (t)", fontsize=14)
    plt.ylabel("Gene Expression Level ($x$)", fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    out_path = os.path.join(out_dir, "realistic_trajectories.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved highly informative trajectory plot to {out_path}")

    # --- Run Inference ---
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON REALISTIC PERTURBED TRAJECTORIES")
    print("="*70)
    
    # Import the inference pipeline
    _proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj_dir not in sys.path:
        sys.path.insert(0, _proj_dir)
    from exp_data_trajectories import run_single_experiment
    
    # Transpose X to shape (N, M) as required by the pipeline
    X_T = X.T
    
    # Run the MCMC inference
    # We use a solid 40 sweeps for a 25-node network to ensure good convergence
    res = run_single_experiment(
        W_true=W, 
        s=X_T, 
        dt_meas=dt_meas, 
        noise_std=np.sqrt(0.1), # Approximation from D_base=0.1
        n_sweeps=40, 
        label="Realistic_Stationary"
    )
    
    print("\n" + "="*70)
    print("INFERENCE RESULTS (REALISTIC TRAJECTORIES)")
    print("="*70)
    print(f"Network Size: N={N}")
    print(f"AUROC: {res['AUROC']:.4f}")
    print(f"AUPRC: {res['AUPRC']:.4f}")
    print(f"F1-Score: {res['prob_metrics']['F1-Score']:.4f}")
    print(f"Precision: {res['prob_metrics']['Precision']:.4f}")
    print(f"Recall (TPR): {res['prob_metrics']['TPR (Recall)']:.4f}")
    print(f"Structural Hamming Distance (SHD): {res['prob_metrics']['SHD']}")
    print("="*70)

if __name__ == '__main__':
    main()
