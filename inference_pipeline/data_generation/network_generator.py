import numpy as np
import graph_tool.all as gt

def generate_scale_free_W(N, m=2, seed=42):
    """
    Generates a scale-free DAG (Price network) and assigns uniform random weights.
    Returns the adjacency matrix W where W[i, j] is the edge from j to i.
    """
    np.random.seed(seed)
    gt.seed_rng(seed)
    
    g = gt.price_network(N, m=m, directed=True)
    
    W = np.zeros((N, N))
    for e in g.edges():
        src = int(e.source())
        tgt = int(e.target())
        
        # Make it a DAG by enforcing source < target
        if src > tgt:
            src, tgt = tgt, src
            
        weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
        W[tgt, src] = weight 
        
    return W

def generate_erdos_renyi_W(N, p=None, m=1, seed=42):
    """
    Generates an Erdos-Renyi DAG and assigns uniform random weights.
    If p is not provided, it is calculated to match the average degree m: p = 2*m / (N - 1).
    Returns the adjacency matrix W where W[i, j] is the edge from j to i.
    """
    np.random.seed(seed)
    
    if p is None:
        p = 2.0 * m / (N - 1) if N > 1 else 0.0
        
    W = np.zeros((N, N))
    for src in range(N):
        for tgt in range(src + 1, N):
            if np.random.rand() < p:
                weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                W[tgt, src] = weight 
                
    return W
