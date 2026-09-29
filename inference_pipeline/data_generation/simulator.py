import numpy as np

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def simulate_realistic_pairs(W, M=200, dt_sim=0.01, dt_sample=0.5, dt_meas=0.05, D_base=0.1, mean_d=0.5, std_d=0.5, mean_V=2.0, std_V=0.5, seed=123, t_window=(0.0, 3.0)):
    N = W.shape[0]
    np.random.seed(seed)
    
    # Heterogeneous parameters on a log scale; homogeneous if std is 0
    if std_d == 0:
        d = np.ones(N) * mean_d
    else:
        d = np.random.lognormal(mean=np.log(mean_d), sigma=std_d, size=N)

    if std_V == 0:
        V = np.ones(N) * mean_V
    else:
        V = np.random.lognormal(mean=np.log(mean_V), sigma=std_V, size=N)
    
    t_start, t_end = t_window
    sample_times = np.arange(t_start, t_end + 1e-9, dt_sample)
    if len(sample_times) == 0:
        sample_times = np.array([t_start])
    K = len(sample_times)
    
    n_trajectories = int(np.ceil(M / K))
    
    T_max = t_end + dt_meas
    n_steps = int(np.round(T_max / dt_sim))
    
    sample_step_indices = [int(np.round(t / dt_sim)) for t in sample_times]
    next_step_indices = [int(np.round((t + dt_meas) / dt_sim)) for t in sample_times]
    
    all_pairs = []
    
    for traj_idx in range(n_trajectories):
        x = np.random.uniform(0.1, 1.0, size=N).astype(np.float64)
        history = {0: x.copy()}
        
        for step in range(1, n_steps + 1):
            drift = -d * x + V * sigmoid(np.dot(W, x))
            noise = np.sqrt(D_base * np.maximum(x, 0.01)) * np.random.normal(0, np.sqrt(dt_sim), size=N)
            x = np.maximum(x + dt_sim * drift + noise, 0.01)
            
            if step in sample_step_indices or step in next_step_indices:
                history[step] = x.copy()
                
        for s_idx, n_idx in zip(sample_step_indices, next_step_indices):
            pair = np.column_stack([history[s_idx], history[n_idx]])
            all_pairs.append(pair)
            if len(all_pairs) == M:
                break
        if len(all_pairs) == M:
            break
            
    out_degrees = (W != 0).sum(axis=0)
    hub_node = np.argmax(out_degrees)
            
    return all_pairs, hub_node, out_degrees, d, V
