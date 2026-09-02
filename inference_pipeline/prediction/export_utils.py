import numpy as np

def export_ensemble_to_bin(save_path="grn_ensemble_results.npz", w_out="W_ensemble.bin", theta_out="theta_ensemble.bin"):
    """
    Loads the npz results from Python and exports them to raw C-order binaries for Julia.
    """
    data = np.load(save_path)
    W = data['W_ensemble'].astype(np.float64) # (N_samples, N, N)
    theta = data['theta_ensemble'].astype(np.float64) # (N_samples, 3)

    W.tofile(w_out)
    theta.tofile(theta_out)
    print(f"Exported ensemble arrays to {w_out} and {theta_out}")

def export_true_to_bin(W_true, theta_true, w_out="W_true.bin", theta_out="theta_true.bin"):
    """
    Exports the True network and True global parameters to raw C-order binaries for Julia.
    """
    W_true.astype(np.float64).tofile(w_out)
    np.array(theta_true, dtype=np.float64).tofile(theta_out)
    print(f"Exported true arrays to {w_out} and {theta_out}")
