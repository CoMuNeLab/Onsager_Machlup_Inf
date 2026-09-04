import numpy as np
from sklearn.linear_model import Lasso

def infer_network(X, t):
    """
    Infers the GRN using Sparse Linear ODE regression (Lasso).
    Models dx/dt = W * x. 
    """
    N = X.shape[1]
    
    # Calculate finite difference derivatives dx/dt
    dt = t[1] - t[0] if len(t) > 1 else 0.5
    dX = (X[1:, :] - X[:-1, :]) / dt
    
    # Input is the state X at time t
    X_in = X[:-1, :]
    
    edge_scores = np.zeros((N, N))
    
    for i in range(N):
        y = dX[:, i]
        
        # Fit sparse linear regression (Lasso)
        # alpha controls sparsity. A small alpha ensures we get some edges.
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_in, y)
        
        # Absolute coefficients act as confidence scores for edges
        edge_scores[i, :] = np.abs(lasso.coef_)
        
    return edge_scores
