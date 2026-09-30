import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def evaluate_topology(W_true, W_ensemble, threshold=0.0, weighted=False):
    """
    Evaluates topological inference (binary edge prediction).
    
    W_true: (N, N) true weight matrix
    W_ensemble: (M, N, N) ensemble of inferred weight matrices
    threshold: absolute weight threshold below which true edges are considered non-edges
    weighted: if True, uses the absolute true weights as sample weights in AUROC/AUPRC
    """
    # Binarize true matrix based on threshold
    true_binary = (np.abs(W_true) > threshold).astype(int)
    
    # Calculate marginal probabilities from ensemble
    ensemble_binary = (np.abs(W_ensemble) > 0).astype(int)
    edge_probabilities = ensemble_binary.mean(axis=0)
    
    # Flatten arrays
    y_true = true_binary.flatten()
    y_scores = edge_probabilities.flatten()
    
    # Sample weights
    if weighted:
        sample_weight = np.abs(W_true).flatten()
        # For negative samples (true weight = 0), we must give them a non-zero weight 
        # so they still count as false positives. We assign them the mean weight of true edges.
        if np.any(true_binary):
            mean_true_weight = np.mean(np.abs(W_true[W_true != 0]))
            sample_weight[y_true == 0] = mean_true_weight
        else:
            sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = None
    
    # Check if there is only one class
    if len(np.unique(y_true)) == 1:
        return {"auroc": np.nan, "auprc": np.nan, "error": "Only one class present in y_true."}
    
    auroc = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)
    auprc = average_precision_score(y_true, y_scores, sample_weight=sample_weight)
    
    baseline = np.mean(y_true)
    auprc_ratio = auprc / baseline if baseline > 0 else np.nan
    auprc_norm = (auprc - baseline) / (1.0 - baseline) if baseline < 1 else np.nan
    
    return {
        "auroc": auroc, 
        "auprc": auprc,
        "auprc_ratio": auprc_ratio,
        "auprc_norm": auprc_norm
    }

def calculate_rbo(l1, l2, p=0.9):
    """
    Calculates Rank-Biased Overlap (RBO) between two lists (l1 and l2).
    p is the parameter determining top-weightedness (0 < p < 1).
    """
    if not l1 or not l2:
        return 0.0
    
    sl1 = set()
    sl2 = set()
    
    depth = max(len(l1), len(l2))
    score = 0.0
    
    for d in range(1, depth + 1):
        if d - 1 < len(l1):
            sl1.add(l1[d-1])
        if d - 1 < len(l2):
            sl2.add(l2[d-1])
            
        overlap = len(sl1.intersection(sl2))
        agreement = overlap / d
        score += (p**(d-1)) * agreement
        
    return (1 - p) * score

def evaluate_weights(W_true, W_ensemble, threshold=0.0):
    """
    Evaluates the quantitative inference of weights.
    Focuses only on true edges (absolute weight > threshold).
    """
    true_edge_mask = np.abs(W_true) > threshold
    
    if not np.any(true_edge_mask):
        return {"mse": np.nan, "pearson": np.nan, "spearman": np.nan, "wasserstein": np.nan, "rbo": np.nan}
        
    true_weights = W_true[true_edge_mask]
    
    # Expected inferred weights
    W_mean = W_ensemble.mean(axis=0)
    inferred_mean_weights = W_mean[true_edge_mask]
    
    # Mean Squared Error
    mse = mean_squared_error(true_weights, inferred_mean_weights)
    
    # Pearson
    if len(np.unique(true_weights)) > 1 and len(np.unique(inferred_mean_weights)) > 1:
        pearson_corr, _ = pearsonr(true_weights, inferred_mean_weights)
    else:
        pearson_corr = np.nan
        
    # Spearman
    spearman_corr, _ = spearmanr(true_weights, inferred_mean_weights)
    
    # Rank-Biased Overlap (RBO)
    # We rank edges by absolute magnitude
    abs_true = np.abs(true_weights)
    abs_inferred = np.abs(inferred_mean_weights)
    
    # Get indices that would sort the arrays in descending order
    rank_true = np.argsort(-abs_true)
    rank_inferred = np.argsort(-abs_inferred)
    
    rbo_score = calculate_rbo(rank_true.tolist(), rank_inferred.tolist(), p=0.9)
    
    # Wasserstein distance (Expected absolute error for delta-function true weights)
    # W_ensemble[:, true_edge_mask] is shape (M, E). For each edge e, we compute W_1 between ensemble and true.
    W_1_per_edge = np.mean(np.abs(W_ensemble[:, true_edge_mask] - true_weights), axis=0)
    mean_wasserstein = np.mean(W_1_per_edge)
    
    return {
        "mse": mse,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "rbo": rbo_score,
        "wasserstein": mean_wasserstein
    }
