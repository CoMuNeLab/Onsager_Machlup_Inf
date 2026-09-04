import os
import sys
import numpy as np

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import simulation functions from our existing notebook/scripts
# Using the updated simulation logic with multiplicative CIR noise
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

def simulate_trajectories(W, T_total, dt, dt_meas, D_base, mean_d, std_d, mean_V, std_V, seed=123):
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
        noise = np.sqrt(D_base * np.maximum(x, 0.01)) * np.random.normal(0, np.sqrt(dt), size=N)
        
        x = x + drift * dt + noise
        x = np.maximum(x, 0.01)
        
        if step % meas_step == 0:
            x_history.append(x.copy())
            t_history.append(t)
            
    return np.array(t_history), np.array(x_history), d, V

def save_dataset(filepath, t, X, W_true, params):
    np.savez_compressed(filepath, t=t, X=X, W_true=W_true, params=params)
    print(f"Saved dataset to {filepath} (Shape: {X.shape})")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    
    N = 20
    T_total = 100.0
    dt = 0.01
    
    # 1. Baseline dataset
    print("Generating baseline dataset...")
    W_baseline = generate_scale_free_W(N, m=2, seed=42)
    t, X, _, _ = simulate_trajectories(W_baseline, T_total, dt, dt_meas=0.5, D_base=0.1, 
                                       mean_d=0.5, std_d=0.5, mean_V=2.0, std_V=0.5, seed=100)
    save_dataset(os.path.join(data_dir, "baseline", "dataset_01.npz"), t, X, W_baseline, {"dt_meas": 0.5, "D_base": 0.1})
    
    # 2. Stress Discretization datasets
    print("Generating stress discretization datasets...")
    for dt_meas in [1.0, 2.0]:
        t, X, _, _ = simulate_trajectories(W_baseline, T_total, dt, dt_meas=dt_meas, D_base=0.1, 
                                           mean_d=0.5, std_d=0.5, mean_V=2.0, std_V=0.5, seed=100)
        save_dataset(os.path.join(data_dir, "stress_discretization", f"dataset_dt_{dt_meas}.npz"), t, X, W_baseline, {"dt_meas": dt_meas, "D_base": 0.1})
        
    # 3. Stress Noise datasets
    print("Generating stress noise datasets...")
    for D_base in [0.5, 1.0]:
        t, X, _, _ = simulate_trajectories(W_baseline, T_total, dt, dt_meas=0.5, D_base=D_base, 
                                           mean_d=0.5, std_d=0.5, mean_V=2.0, std_V=0.5, seed=100)
        save_dataset(os.path.join(data_dir, "stress_noise", f"dataset_D_{D_base}.npz"), t, X, W_baseline, {"dt_meas": 0.5, "D_base": D_base})
        
    print("All datasets generated successfully!")
