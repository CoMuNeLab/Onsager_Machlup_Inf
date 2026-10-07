import numpy as np
from sklearn.ensemble import RandomForestRegressor

def infer_network(X, t):
    """
    Infers the GRN using Random Forest Regression (dynGENIE3 approximation).
    Predicts X(t+1) from X(t). Feature importances become edge scores.
    """
    N = X.shape[1]
    
    # Create input-output pairs
    X_in = X[:-1, :]
    X_out = X[1:, :]
    
    edge_scores = np.zeros((N, N))
    
    for i in range(N):
        y = X_out[:, i]
        
        # Fit Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_in, y)
        
        # Feature importances are the edge weights (from all j to i)
        edge_scores[i, :] = rf.feature_importances_
        
    return edge_scores
