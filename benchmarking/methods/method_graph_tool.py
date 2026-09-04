import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import graph_tool.all as gt
from gene_expr import GeneExprBlockState

def infer_network(X, t, n_sweeps=30):
    """
    Infers the GRN using the graph-tool SDE method.
    Returns: Edge score matrix of shape (N, N).
    """
    N = X.shape[1]
    M = X.shape[0]
    dt_meas = t[1] - t[0] if len(t) > 1 else 0.5
    
    # Format for graph-tool
    s = X.T
    
    # Initialize empty graph state
    state = GeneExprBlockState(s, dt=dt_meas)
    
    # Run MCMC
    best_S = state.entropy()
    best_graph = state.get_graph().copy()
    
    for sweep in range(n_sweeps):
        state.mcmc_sweep(niter=10)
        S_cur = state.entropy()
        if S_cur < best_S:
            best_S = S_cur
            best_graph = state.get_graph().copy()
            
    # Extract edges
    edge_scores = np.zeros((N, N))
    
    try:
        weight_ep = state.x
    except AttributeError:
        ep_keys = list(best_graph.ep.keys())
        weight_ep = best_graph.ep[ep_keys[0]] if ep_keys else None
        
    for e in best_graph.edges():
        src, tgt = int(e.source()), int(e.target())
        if src != tgt:
            w = 1.0
            if weight_ep is not None:
                val = weight_ep[e]
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    w = abs(float(val[0]))
                else:
                    w = abs(float(val))
            # We use absolute weight as the confidence score
            edge_scores[tgt, src] = w
            
    return edge_scores
