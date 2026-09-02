import numpy as np
import graph_tool.all as gt

def generate_scale_free_W(N, m=2, seed=42):
    np.random.seed(seed)
    gt.seed_rng(seed)
    # Generate scale-free DAG (Price network)
    g = gt.price_network(N, m=m, directed=True)
    
    W = np.zeros((N, N))
    for e in g.edges():
        src = int(e.source())
        tgt = int(e.target())
        
        # Make it a DAG by enforcing source < target
        if src > tgt:
            src, tgt = tgt, src
            
        weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
        W[tgt, src] = weight # W[i, j] is edge from j to i
        
    return W

N = 50
m = 1
seed_network = 42
mean_d = 0.5
mean_V = 2.0
D_base = 0.02

W_true = generate_scale_free_W(N, m, seed=seed_network)
theta_true = np.array([np.log(mean_d), np.log(mean_V), np.log(D_base)])

W_true.astype(np.float64).tofile('W_true.bin')
theta_true.astype(np.float64).tofile('theta_true.bin')

print("Exported W_true.bin and theta_true.bin")
