import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# We reuse the simulation function from our experiment script
_proj_dir = os.path.dirname(os.path.abspath(__file__))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

from exp_information_content import generate_network, simulate_trajectory

out_dir = "/home/francesco/.gemini/antigravity/brain/bfdc093e-0921-4ff2-a18c-45b223faec92"

def main():
    # Parameters that yielded the highest stationary information
    N = 20 # Pick a moderate size
    topo = 'ER'
    d = 0.5
    V = 5.0
    noise = 0.05
    dt_meas = 0.05
    
    np.random.seed(42) # Reproducibility
    W = generate_network(N, topo)
    
    # Simulate a longer trajectory
    tau = 1.0 / d
    T_transient_end = 5.0 * tau
    T_total = T_transient_end + 60.0
    
    traj = simulate_trajectory(W, T_total=T_total, dt_sim=0.005, dt_meas=dt_meas, 
                               d=d, V=V, noise_std=noise)
    
    # Plot the stationary regime
    idx_trans = int(T_transient_end / dt_meas)
    time = np.arange(idx_trans, len(traj)) * dt_meas
    traj_stat = traj[idx_trans:, :]
    
    plt.figure(figsize=(14, 6))
    
    # Plot 5 nodes with the highest variance to actually show the limit cycle
    variances = np.var(traj_stat, axis=0)
    top_nodes = np.argsort(variances)[-5:]
    
    for i in top_nodes:
        plt.plot(time, traj_stat[:, i], label=f'Node {i}')
        
    plt.title(f"Informative Stationary Regime (V={V}, d={d}, topo={topo})")
    plt.xlabel("Time (s)")
    plt.ylabel("Expression Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_informative_stationary.png"), dpi=150)
    print("Saved trajectory plot.")

if __name__ == "__main__":
    main()
