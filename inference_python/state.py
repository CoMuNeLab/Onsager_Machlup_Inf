import numpy as np
from inference_python.likelihood import node_log_likelihood, compute_m
from inference_python.prior import compute_mdl_prior

class GRNInferenceState:
    """
    State object for GRN Inference tracking network topology (W),
    weight categories (z), counts (m_k), and evaluating Description Length (Entropy).
    """
    def __init__(self, s, dt, theta=None):
        """
        Parameters:
        - s: List of M arrays, each of shape (N, 2). Column 0 is x(t), Column 1 is x(t+dt).
        - dt: Measurement time step.
        - theta: Initial node parameters.
        """
        self.dt = dt
        
        if isinstance(s, list):
            # Convert to numpy arrays
            # X_t shape: (N, M)
            self.X_t = np.stack([p[:, 0] for p in s], axis=1)
            # X_next shape: (N, M)
            self.X_next = np.stack([p[:, 1] for p in s], axis=1)
        else:
            raise ValueError("s should be a list of (N, 2) arrays as in graph-tool pipeline")
            
        self.N, self.M = self.X_t.shape
        
        if theta is None: # Default initialization
            self.theta = np.zeros((self.N, 3))
            self.theta[:, 0] = np.log(0.5)  # gamma
            self.theta[:, 1] = np.log(2.0)  # V
            self.theta[:, 2] = np.log(0.05) # D
        else:
            self.theta = np.array(theta)
            
        # Network state, initially empty
        self.W = np.zeros((self.N, self.N))
        self.E = 0
        
        # Category state, initially empty
        self.K = 0
        self.z = [] # list of floats
        self.m_counts = [] # list of ints
        
        # Mapping from edge tuple (tgt, src) to category index k
        # Note: graph-tool convention: edge from src to tgt is W[tgt, src]
        self.edge_to_cat = {} 
        
        # Cache of node log-likelihoods
        self.node_ll = np.zeros(self.N)
        self._update_all_likelihoods()

    def _update_all_likelihoods(self):
        """Recomputes likelihood for all nodes."""
        for i in range(self.N):
            self.node_ll[i] = self._calc_node_ll(i)

    def _calc_node_ll(self, i, W_i=None, theta_i=None):
        """Calculates likelihood for a single node i given potential new W_i or theta_i."""
        if W_i is None:
            W_i = self.W[i, :]
        if theta_i is None:
            theta_i = self.theta[i, :]
            
        m = compute_m(W_i, self.X_t)
        return node_log_likelihood(self.X_t[i, :], self.X_next[i, :], m, theta_i, self.dt)

    def get_entropy(self):
        """
        Total Description Length (Entropy) = -log P(X|W, Theta) - log P(W, z)
        """
        ll = np.sum(self.node_ll)
        prior = compute_mdl_prior(self.N, self.E, self.K, self.m_counts, self.z)
        return -ll + prior

    def get_delta_entropy_edge(self, tgt, src, new_val):
        """
        Computes the change in entropy if we were to change W[tgt, src] to new_val.
        Does not actually change the state.
        
        For the prior, we need to determine the implied changes to E, K, m_counts, and z.
        This is a bit complex, so we will create a temporary copy of the prior variables.
        """
        old_val = self.W[tgt, src]
        if old_val == new_val:
            return 0.0
            
        # 1. Delta Likelihood
        W_i_new = self.W[tgt, :].copy()
        W_i_new[src] = new_val
        new_node_ll = self._calc_node_ll(tgt, W_i_new)
        delta_ll = new_node_ll - self.node_ll[tgt]
        
        # 2. Delta Prior
        new_E, new_K, new_m_counts, new_z = self._simulate_prior_update_edge(tgt, src, new_val)
        old_prior = compute_mdl_prior(self.N, self.E, self.K, self.m_counts, self.z)
        new_prior = compute_mdl_prior(self.N, new_E, new_K, new_m_counts, new_z)
        delta_prior = new_prior - old_prior
        
        delta_entropy = -delta_ll + delta_prior
        return delta_entropy
        
    def _simulate_prior_update_edge(self, tgt, src, new_val):
        """Helper to compute what the prior variables would be if an edge changed."""
        new_E = self.E
        new_m_counts = list(self.m_counts)
        new_z = list(self.z)
        
        # Remove old edge contribution
        if (tgt, src) in self.edge_to_cat:
            old_k = self.edge_to_cat[(tgt, src)]
            new_m_counts[old_k] -= 1
            new_E -= 1
            
        # Add new edge contribution
        if new_val != 0.0:
            new_E += 1
            # Check if new_val exists in current categories
            # Use small tolerance for float equality
            k_idx = next((i for i, z_val in enumerate(new_z) if abs(z_val - new_val) < 1e-6), None)
            if k_idx is not None:
                new_m_counts[k_idx] += 1
            else:
                new_z.append(new_val)
                new_m_counts.append(1)
                
        # Clean up empty categories
        clean_z = []
        clean_m = []
        for z_val, m in zip(new_z, new_m_counts):
            if m > 0:
                clean_z.append(z_val)
                clean_m.append(m)
                
        new_K = len(clean_z)
        return new_E, new_K, clean_m, clean_z

    def apply_edge_update(self, tgt, src, new_val):
        """Applies the edge update to the actual state."""
        if self.W[tgt, src] == new_val:
            return
            
        # Update Topology
        self.W[tgt, src] = new_val
        
        # Update Node Likelihood
        self.node_ll[tgt] = self._calc_node_ll(tgt)
        
        # Update Categories
        if (tgt, src) in self.edge_to_cat:
            old_k = self.edge_to_cat[(tgt, src)]
            self.m_counts[old_k] -= 1
            self.E -= 1
            del self.edge_to_cat[(tgt, src)]
            
        if new_val != 0.0:
            self.E += 1
            k_idx = next((i for i, z_val in enumerate(self.z) if abs(z_val - new_val) < 1e-6), None)
            if k_idx is not None:
                self.m_counts[k_idx] += 1
                self.edge_to_cat[(tgt, src)] = k_idx
            else:
                self.z.append(new_val)
                self.m_counts.append(1)
                self.edge_to_cat[(tgt, src)] = len(self.z) - 1
                self.K += 1
                
        # Garbage collect empty categories and remap
        self._gc_categories()
        
    def _gc_categories(self):
        """Removes empty categories and remaps edge_to_cat."""
        if 0 not in self.m_counts:
            return
            
        old_to_new_k = {}
        clean_z = []
        clean_m = []
        
        new_k = 0
        for old_k, m in enumerate(self.m_counts):
            if m > 0:
                clean_z.append(self.z[old_k])
                clean_m.append(m)
                old_to_new_k[old_k] = new_k
                new_k += 1
                
        self.z = clean_z
        self.m_counts = clean_m
        self.K = len(self.z)
        
        # Remap edges
        for edge in list(self.edge_to_cat.keys()):
            old_k = self.edge_to_cat[edge]
            if old_k in old_to_new_k:
                self.edge_to_cat[edge] = old_to_new_k[old_k]
            else:
                del self.edge_to_cat[edge]

    def get_delta_entropy_theta(self, i, new_theta):
        """
        Computes the change in entropy if we change the parameters of node i.
        Since we assume a uniform prior over theta within its bounds,
        the delta entropy is exactly the negative delta likelihood.
        """
        new_node_ll = self._calc_node_ll(i, theta_i=new_theta)
        delta_ll = new_node_ll - self.node_ll[i]
        return -delta_ll

    def apply_theta_update(self, i, new_theta):
        """Applies the parameter update to node i."""
        self.theta[i, :] = new_theta
        self.node_ll[i] = self._calc_node_ll(i)

