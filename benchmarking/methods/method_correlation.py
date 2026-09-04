import numpy as np
import pandas as pd

def infer_network(X, t):
    """
    Infers the GRN using Time-Delayed Pearson Correlation.
    Calculates the correlation between X_j(t) and X_i(t+1).
    """
    N = X.shape[1]
    
    X_in = X[:-1, :]
    X_out = X[1:, :]
    
    edge_scores = np.zeros((N, N))
    
    df_in = pd.DataFrame(X_in)
    df_out = pd.DataFrame(X_out)
    
    for i in range(N):
        # Target at t+1
        target_series = df_out.iloc[:, i]
        
        # Correlate with all regulators at t
        corrs = df_in.corrwith(target_series, method='pearson')
        
        # Absolute correlation acts as confidence score
        edge_scores[i, :] = corrs.abs().values
        
    return edge_scores
