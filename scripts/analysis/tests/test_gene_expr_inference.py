import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
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

    test_simulate_grn_sde_pairs_invariance()

def test_simulate_grn_sde_pairs_invariance():
    from inference_pipeline.data_generation.network_generator import generate_scale_free_W
    from inference_pipeline.data_generation.simulator import simulate_realistic_pairs
    print("\n" + "=" * 60)
    print("Testing State-Velocity Pair Sampling (`simulate_realistic_pairs`) & MCMC")
    print("=" * 60)
    W = generate_scale_free_W(10, m=1, seed=42)
    pairs, hub, out_degs, d_true, V_true = simulate_realistic_pairs(W, M=30, dt_sim=0.001, dt_sample=0.5, dt_meas=0.1, seed=42)

    assert len(pairs) == 30, f"Expected 30 pairs, got {len(pairs)}"
    assert pairs[0].shape == (10, 2), f"Expected shape (10, 2), got {pairs[0].shape}"
    print("✓ Pair generation verified: 30 pairs of shape (10, 2) generated successfully!")

    state = GeneExprBlockState(pairs, dt=0.1)
    S_init = state.entropy()
    print(f"✓ GeneExprBlockState instantiated with list of pairs. Initial entropy = {S_init:.4f}")
    dS, nmoves = state.mcmc_sweep(niter=5)
    print(f"✓ MCMC sweep completed on pairs: dS = {dS:.4f}, accepted moves = {nmoves}")
    print("SUCCESS! All pair simulation and MCMC tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
