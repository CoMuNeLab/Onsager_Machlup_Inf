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
