import numpy as np

# Load npz
data = np.load('grn_ensemble_results.npz')

# Extract
W = data['W_ensemble'].astype(np.float64) # (N_samples, N, N)
theta = data['theta_ensemble'].astype(np.float64) # (N_samples, 3)

# Write to raw binary
W.tofile('W_ensemble.bin')
theta.tofile('theta_ensemble.bin')

print("W shape:", W.shape)
print("theta shape:", theta.shape)
