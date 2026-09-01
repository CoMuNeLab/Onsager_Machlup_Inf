import numpy as np
import scipy.special

def log_binom(n, k):
    """Computes log(n choose k) using scipy gammaln."""
    if k < 0 or k > n:
        return -np.inf
    return scipy.special.gammaln(n + 1) - scipy.special.gammaln(k + 1) - scipy.special.gammaln(n - k + 1)

def compute_mdl_prior(N, E, K, m_counts, z_values, lambda_val=1.0, delta=1e-8):
    """
    Computes the Minimum Description Length (MDL) prior penalty: -log P(W, z | lambda, Delta).
    
    Parameters:
    - N: Number of nodes
    - E: Number of edges
    - K: Number of categories
    - m_counts: Array or list of edge counts per category (length K)
    - z_values: Array or list of category values (length K)
    - lambda_val: Scale parameter for Laplace prior
    - delta: Precision parameter for Laplace prior
    
    Returns:
    - prior_penalty: -log P(W)
    """
    if E == 0 and K == 0:
        return 0.0 # Base case, no penalty for empty graph? 
                   # Wait, for empty graph E=0, K=0, the formula needs careful limits.
                   # Let's handle E=0 properly.
    
    N_pairs = N * (N - 1) # Total possible directed edges in a graph with N nodes (excluding self-loops)
    
    if E == 0:
        # If E=0, K must be 0.
        # -log P(W) = log(N_pairs + 1)
        return np.log(N_pairs + 1.0)
        
    if K == 0 or K > E:
        return np.inf

    # -log(prod m_k!)
    term1 = -np.sum([scipy.special.gammaln(m + 1) for m in m_counts])
    
    # lambda * sum |z_k|
    term2 = lambda_val * np.sum(np.abs(z_values))
    
    # - K * log(e^(lambda * delta) - 1)
    # Using numerical approximation for very small delta: e^(x) - 1 approx x
    if lambda_val * delta < 1e-4:
        log_e_minus_1 = np.log(lambda_val * delta)
    else:
        log_e_minus_1 = np.log(np.exp(lambda_val * delta) - 1.0)
    term3 = -K * log_e_minus_1
    
    # log(E!)
    term4 = scipy.special.gammaln(E + 1)
    
    # log binom(E-1, K-1)
    term5 = log_binom(E - 1, K - 1)
    
    # K * log(2)
    term6 = K * np.log(2.0)
    
    # log(max(E, 1))
    term7 = np.log(max(E, 1))
    
    # log binom(N_pairs, E)
    term8 = log_binom(N_pairs, E)
    
    # log(N_pairs + 1)
    term9 = np.log(N_pairs + 1.0)
    
    total_penalty = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    return total_penalty
