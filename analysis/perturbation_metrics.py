import numpy as np
from scipy.stats import pearsonr, spearmanr, gaussian_kde

def evaluate_shift_correlation(true_wt, true_pert, ensemble_wt, ensemble_pert, ko_idx):
    """
    Evaluates the correlation of the state shift (Delta X) caused by perturbation.
    
    true_wt: (N,) true wild-type steady state
    true_pert: (N,) true perturbed steady state
    ensemble_wt: (M, N) ensemble wild-type steady states
    ensemble_pert: (M, N) ensemble perturbed steady states
    ko_idx: index of the knocked-out node (excluded from correlation)
    """
    # True shift
    delta_true = true_pert - true_wt
    
    # Ensemble shift (properly paired)
    # Assuming ensemble_pert and ensemble_wt are ordered pairs from the same matrices
    delta_ensemble = ensemble_pert - ensemble_wt
    mean_delta_ensemble = delta_ensemble.mean(axis=0)
    
    # Mask out the knocked-out node
    mask = np.ones(len(delta_true), dtype=bool)
    mask[ko_idx] = False
    
    dt = delta_true[mask]
    de = mean_delta_ensemble[mask]
    
    if len(np.unique(dt)) > 1 and len(np.unique(de)) > 1:
        pearson_corr, _ = pearsonr(dt, de)
    else:
        pearson_corr = np.nan
        
    spearman_corr, _ = spearmanr(dt, de)
    
    return {
        "pearson_shift": pearson_corr,
        "spearman_shift": spearman_corr
    }

def evaluate_calibration(true_pert, ensemble_pert, ko_idx, alpha=0.05):
    """
    Evaluates if the true perturbed state falls within the ensemble's confidence intervals.
    """
    mask = np.ones(len(true_pert), dtype=bool)
    mask[ko_idx] = False
    
    # Calculate bounds over the ensemble distribution
    lower_bound = np.percentile(ensemble_pert, (alpha / 2) * 100, axis=0)
    upper_bound = np.percentile(ensemble_pert, (1 - alpha / 2) * 100, axis=0)
    
    within_bounds = (true_pert >= lower_bound) & (true_pert <= upper_bound)
    fraction_within = np.mean(within_bounds[mask])
    
    return {
        "fraction_within_ci": fraction_within
    }

def evaluate_multimodal_kde(true_pert, ensemble_pert, ko_idx):
    """
    Fits a KDE to each node's ensemble predictions and computes the log-likelihood 
    of the true state under this empirical distribution to handle multimodality.
    """
    N_nodes = len(true_pert)
    log_likelihoods = []
    
    for i in range(N_nodes):
        if i == ko_idx:
            continue
            
        data = ensemble_pert[:, i]
        
        # Check if the ensemble is completely certain (zero variance)
        if np.var(data) < 1e-10:
            if np.abs(true_pert[i] - data[0]) < 1e-5:
                # Perfect match, infinite log-likelihood conceptually, let's cap it
                log_likelihoods.append(10.0) 
            else:
                # Confident but wrong, low log-likelihood
                log_likelihoods.append(-10.0)
            continue
            
        try:
            kde = gaussian_kde(data)
            # Evaluate logpdf at the true state
            ll = kde.logpdf(true_pert[i])[0]
            log_likelihoods.append(ll)
        except Exception as e:
            # Fallback if KDE fails (e.g. singular matrix)
            log_likelihoods.append(np.nan)
            
    return {
        "mean_log_likelihood": np.nanmean(log_likelihoods)
    }
