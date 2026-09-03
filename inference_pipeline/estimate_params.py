import numpy as np
import pandas as pd

def estimate_parameters():
    # Load the DREAM4 time series data
    df = pd.read_csv('/tmp/BSBL-TSK-FS_code/insilico_size10_1_timeseries.tsv', sep='\t')
    data = df.values
    
    n_perturbations = 5
    timepoints = 21
    
    dx = []
    x_vals = []
    
    for i in range(n_perturbations):
        start_idx = i * timepoints
        for t in range(timepoints - 1):
            x_t = data[start_idx + t]
            x_t_next = data[start_idx + t + 1]
            
            x_vals.append(x_t)
            # We normalized dt to 1.0 in our inference script
            dt = 1.0 
            dx.append((x_t_next - x_t) / dt)
            
    dx = np.array(dx)       # shape: (100, 10)
    x_vals = np.array(x_vals) # shape: (100, 10)
    
    # 1. Estimate d (degradation rate)
    # When dx/dt is very negative, production is likely ~0, so dx/dt ~ -d * x
    # We look at the most negative relative changes (dx/dt) / x
    relative_changes = dx / np.maximum(x_vals, 0.01)
    
    # Take the 5th percentile of relative changes to represent pure degradation drops
    d_estimates = -np.percentile(relative_changes, 5, axis=0)
    d_global = np.median(d_estimates)
    # Ensure it's strictly positive
    d_global = max(d_global, 0.1)
    
    # 2. Estimate V (maximal production rate)
    # When dx/dt is very positive, dx/dt ~ -d*x + V*sigmoid(..). Max sigmoid is 1, so V ~ dx/dt + d*x
    # We look at the 95th percentile of absolute jumps
    production_terms = dx + d_global * x_vals
    V_estimates = np.percentile(production_terms, 95, axis=0)
    V_global = np.median(V_estimates)
    V_global = max(V_global, 0.1)
    
    # 3. Estimate D_base (noise parameter)
    # We look at points where dx/dt is close to 0 (steady state). 
    # In Euler-Maruyama: Var(dx) = D_base * x * dt
    # So D_base = Var(dx) / (x * dt)
    # Let's just look at the empirical variance of the residuals
    # Since we don't know the true W, we can estimate variance by looking at the small fluctuations
    # Take the standard deviation of dx/dt overall as a rough ceiling for noise
    overall_variance = np.var(dx, axis=0)
    mean_x = np.mean(x_vals, axis=0)
    D_base_estimates = overall_variance / mean_x
    # Since variance includes signal (dynamics), we assume noise is e.g. 10% of total variance
    D_base_global = np.median(D_base_estimates) * 0.10
    D_base_global = max(D_base_global, 0.001)

    print(f"Estimated d (degradation): {d_global:.4f}")
    print(f"Estimated V (production) : {V_global:.4f}")
    print(f"Estimated D_base (noise) : {D_base_global:.4f}")
    
    print("\nCorresponding log parameters for initial_theta:")
    print(f"log(d)      : {np.log(d_global):.4f}")
    print(f"log(V)      : {np.log(V_global):.4f}")
    print(f"log(D_base) : {np.log(D_base_global):.4f}")

if __name__ == '__main__':
    estimate_parameters()
