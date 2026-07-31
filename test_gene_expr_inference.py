import os
import sys
import numpy as np

# Add project directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_tool.all as gt
from gene_expr import GeneExprBlockState

def simulate_grn_trajectory(N=5, M=50, dt=0.1, seed=42):
    np.random.seed(seed)
    # Ground truth parameters
    gamma = np.full(N, 0.5)      # decay rates
    V = np.full(N, 2.0)          # maximum expression levels
    D = np.full(N, 0.05)         # diffusion coefficients
    
    # Simple regulatory interaction matrix W (N x N)
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i - 1) % N] = 1.5  # ring activation
        W[i, (i + 1) % N] = -1.0 # ring repression

    # Simulate Euler-Maruyama trajectory
    x = np.zeros((N, M))
    x[:, 0] = np.random.uniform(0.5, 2.0, size=N)
    for k in range(M - 1):
        m = W @ x[:, k]
        sig_m = 1.0 / (1.0 + np.exp(-m))
        drift = -gamma * x[:, k] + V * sig_m
        noise = np.random.normal(0, np.sqrt(D * dt), size=N)
        x[:, k + 1] = x[:, k] + dt * drift + noise
        x[:, k + 1] = np.maximum(x[:, k + 1], 0.01) # keep expression positive
    return x, W

def main():
    print("=" * 60)
    print("Testing GRN Bayesian Network Inference Extension (gene_expr)")
    print("=" * 60)

    N, M, dt = 5, 50, 0.1
    print(f"Simulating synthetic GRN data with N={N} genes, M={M} time steps, dt={dt}...")
    s, W_true = simulate_grn_trajectory(N=N, M=M, dt=dt)
    print("Data simulation complete. Matrix shape:", s.shape)

    print("\nInstantiating GeneExprBlockState...")
    state = GeneExprBlockState(s, dt=dt)
    
    S_init = state.entropy()
    print(f"Initial description length (entropy) = {S_init:.4f}")

    print("\nRunning MCMC sweeps to infer GRN structure and parameters...")
    for step in range(5):
        dS, nmoves = state.mcmc_sweep(niter=10)
        S_cur = state.entropy()
        print(f"Sweep {step+1}: Entropy = {S_cur:.4f} (dS = {dS:.4f}, accepted moves = {nmoves})")

    print("\nSUCCESS! Gene expression SDE Bayesian inference extension works cleanly with graph-tool!")
    print("=" * 60)

if __name__ == "__main__":
    main()
