import numpy as np
from inference_python.likelihood import node_log_likelihood, compute_m
from inference_python.prior import compute_mdl_prior, compute_mdl_theta_prior

class GRNInferenceState:
    """
    State object for GRN Inference tracking network topology (W),
    weight categories (z), counts (m_k), and evaluating Description Length (Entropy).
    Now also tracks categorical node parameters (theta).
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
            self.X_t = np.stack([p[:, 0] for p in s], axis=1)
            self.X_next = np.stack([p[:, 1] for p in s], axis=1)
        elif isinstance(s, np.ndarray):
            self.X_t = s
            self.X_next = dt 
            self.dt = theta
        else:
            raise ValueError("s should be a list of (N, 2) arrays as in graph-tool pipeline")
            
        self.N, self.M = self.X_t.shape
        
        # Network state, initially empty
        self.W = np.zeros((self.N, self.N))
        self.E = 0
        
        # Edge Category state
        self.K = 0
        self.z = [] 
        self.m_counts = []
        self.edge_to_cat = {} 
        
        # Theta Parameter state
        self.theta = np.zeros((self.N, 3))
        self.theta_K = [0, 0, 0]
        self.theta_u = [[], [], []]
        self.theta_n = [[], [], []]
        self.node_to_theta_cat = [{}, {}, {}]
        
        self.node_ll = np.zeros(self.N)

        if not isinstance(s, np.ndarray):
            # Initialize Theta only on the main constructor path
            if theta is None:
                init_vals = [np.log(0.5), np.log(2.0), np.log(0.05)]
            else:
                init_vals = theta[0] # assume homogeneous start
                    
            for p in range(3):
                val = init_vals[p]
                self.theta[:, p] = val
                self.theta_u[p].append(val)
                self.theta_n[p].append(self.N)
                self.theta_K[p] = 1
                for i in range(self.N):
                    self.node_to_theta_cat[p][i] = 0
            
            self._update_all_likelihoods()

    def clone(self):
        """Creates a deep copy of the state, but shares X_t and X_next."""
        import copy
        new_state = GRNInferenceState(self.X_t, self.X_next, self.dt)
        new_state.W = self.W.copy()
        new_state.E = self.E
        new_state.K = self.K
        new_state.z = list(self.z)
        new_state.m_counts = list(self.m_counts)
        new_state.edge_to_cat = dict(self.edge_to_cat)
        new_state.theta = self.theta.copy()
        new_state.theta_K = list(self.theta_K)
        new_state.theta_u = [list(u) for u in self.theta_u]
        new_state.theta_n = [list(n) for n in self.theta_n]
        new_state.node_to_theta_cat = [dict(d) for d in self.node_to_theta_cat]
        new_state.node_ll = self.node_ll.copy()
        return new_state

    def _update_all_likelihoods(self):
        for i in range(self.N):
            self.node_ll[i] = self._calc_node_ll(i)

    def _calc_node_ll(self, i, W_i=None, theta_i=None):
        if W_i is None:
            W_i = self.W[i, :]
        if theta_i is None:
            theta_i = self.theta[i, :]
            
        m = compute_m(W_i, self.X_t)
        return node_log_likelihood(self.X_t[i, :], self.X_next[i, :], m, theta_i, self.dt)

    def get_entropy(self):
        """
        Total Description Length (Entropy) = -log P(X|W, Theta) - log P(W, z) - log P(Theta, u)
        """
        ll = np.sum(self.node_ll)
        prior_W = compute_mdl_prior(self.N, self.E, self.K, self.m_counts, self.z)
        prior_theta = 0.0
        for p in range(3):
            prior_theta += compute_mdl_theta_prior(self.N, self.theta_K[p], self.theta_n[p], self.theta_u[p])
        return -ll + prior_W + prior_theta

    def get_delta_entropy_edge(self, tgt, src, new_val):
        old_val = self.W[tgt, src]
        if old_val == new_val:
            return 0.0
            
        W_i_new = self.W[tgt, :].copy()
        W_i_new[src] = new_val
        new_node_ll = self._calc_node_ll(tgt, W_i_new)
        delta_ll = new_node_ll - self.node_ll[tgt]
        
        new_E, new_K, new_m_counts, new_z = self._simulate_prior_update_edge(tgt, src, new_val)
        old_prior = compute_mdl_prior(self.N, self.E, self.K, self.m_counts, self.z)
        new_prior = compute_mdl_prior(self.N, new_E, new_K, new_m_counts, new_z)
        delta_prior = new_prior - old_prior
        
        delta_entropy = -delta_ll + delta_prior
        return delta_entropy
        
    def _simulate_prior_update_edge(self, tgt, src, new_val):
        new_E = self.E
        new_m_counts = list(self.m_counts)
        new_z = list(self.z)
        
        if (tgt, src) in self.edge_to_cat:
            old_k = self.edge_to_cat[(tgt, src)]
            new_m_counts[old_k] -= 1
            new_E -= 1
            
        if new_val != 0.0:
            new_E += 1
            k_idx = next((i for i, z_val in enumerate(new_z) if abs(z_val - new_val) < 1e-6), None)
            if k_idx is not None:
                new_m_counts[k_idx] += 1
            else:
                new_z.append(new_val)
                new_m_counts.append(1)
                
        clean_z = []
        clean_m = []
        for z_val, m in zip(new_z, new_m_counts):
            if m > 0:
                clean_z.append(z_val)
                clean_m.append(m)
                
        new_K = len(clean_z)
        return new_E, new_K, clean_m, clean_z

    def get_delta_entropy_edge_subset(self, edges, new_val):
        """Evaluates changing a subset of edges to a single new value."""
        if not edges:
            return 0.0
            
        affected_nodes = set([tgt for tgt, src in edges])
        
        # 1. Delta Likelihood
        old_ll_sum = sum(self.node_ll[tgt] for tgt in affected_nodes)
        
        W_new = self.W.copy()
        for tgt, src in edges:
            W_new[tgt, src] = new_val
            
        new_ll_sum = 0.0
        for tgt in affected_nodes:
            new_ll_sum += self._calc_node_ll(tgt, W_i=W_new[tgt, :])
            
        delta_ll = new_ll_sum - old_ll_sum
        
        # 2. Delta Prior
        new_E = self.E
        new_m_counts = list(self.m_counts)
        new_z = list(self.z)
        
        for tgt, src in edges:
            if (tgt, src) in self.edge_to_cat:
                old_k = self.edge_to_cat[(tgt, src)]
                new_m_counts[old_k] -= 1
                new_E -= 1
                
            if new_val != 0.0:
                new_E += 1
                k_idx = next((i for i, z_val in enumerate(new_z) if abs(z_val - new_val) < 1e-6), None)
                if k_idx is not None:
                    new_m_counts[k_idx] += 1
                else:
                    new_z.append(new_val)
                    new_m_counts.append(1)
                    
        clean_z = []
        clean_m = []
        for z_val, m in zip(new_z, new_m_counts):
            if m > 0:
                clean_z.append(z_val)
                clean_m.append(m)
                
        new_K = len(clean_z)
        
        old_prior = compute_mdl_prior(self.N, self.E, self.K, self.m_counts, self.z)
        new_prior = compute_mdl_prior(self.N, new_E, new_K, clean_m, clean_z)
        delta_prior = new_prior - old_prior
        
        return -delta_ll + delta_prior

    def apply_edge_update(self, tgt, src, new_val):
        if self.W[tgt, src] == new_val:
            return
            
        self.W[tgt, src] = new_val
        self.node_ll[tgt] = self._calc_node_ll(tgt)
        
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
                
        self._gc_edge_categories()
        
    def apply_edge_update_subset(self, edges, new_val):
        """Applies an update to a subset of edges simultaneously."""
        for tgt, src in edges:
            self.W[tgt, src] = new_val
            if (tgt, src) in self.edge_to_cat:
                old_k = self.edge_to_cat[(tgt, src)]
                self.m_counts[old_k] -= 1
                self.E -= 1
                del self.edge_to_cat[(tgt, src)]
                
        if new_val != 0.0:
            k_idx = next((i for i, z_val in enumerate(self.z) if abs(z_val - new_val) < 1e-6), None)
            if k_idx is None:
                self.z.append(new_val)
                self.m_counts.append(0)
                self.K += 1
                k_idx = len(self.z) - 1
                
            for tgt, src in edges:
                self.E += 1
                self.m_counts[k_idx] += 1
                self.edge_to_cat[(tgt, src)] = k_idx
                
        affected_nodes = set([tgt for tgt, src in edges])
        for tgt in affected_nodes:
            self.node_ll[tgt] = self._calc_node_ll(tgt)
            
        self._gc_edge_categories()

    def _gc_edge_categories(self):
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
        
        for edge in list(self.edge_to_cat.keys()):
            old_k = self.edge_to_cat[edge]
            if old_k in old_to_new_k:
                self.edge_to_cat[edge] = old_to_new_k[old_k]
            else:
                del self.edge_to_cat[edge]

    # --- THETA CATEGORY METHODS ---

    def get_delta_entropy_theta_category(self, i, p_idx, new_val):
        old_val = self.theta[i, p_idx]
        if old_val == new_val:
            return 0.0
            
        theta_i_new = self.theta[i, :].copy()
        theta_i_new[p_idx] = new_val
        new_node_ll = self._calc_node_ll(i, theta_i=theta_i_new)
        delta_ll = new_node_ll - self.node_ll[i]
        
        new_K, new_n_counts, new_u = self._simulate_prior_update_theta(i, p_idx, new_val)
        old_prior = compute_mdl_theta_prior(self.N, self.theta_K[p_idx], self.theta_n[p_idx], self.theta_u[p_idx])
        new_prior = compute_mdl_theta_prior(self.N, new_K, new_n_counts, new_u)
        delta_prior = new_prior - old_prior
        
        return -delta_ll + delta_prior

    def _simulate_prior_update_theta(self, i, p_idx, new_val):
        new_n_counts = list(self.theta_n[p_idx])
        new_u = list(self.theta_u[p_idx])
        
        old_k = self.node_to_theta_cat[p_idx][i]
        new_n_counts[old_k] -= 1
            
        k_idx = next((k for k, u_val in enumerate(new_u) if abs(u_val - new_val) < 1e-6), None)
        if k_idx is not None:
            new_n_counts[k_idx] += 1
        else:
            new_u.append(new_val)
            new_n_counts.append(1)
                
        clean_u = []
        clean_n = []
        for u_val, n in zip(new_u, new_n_counts):
            if n > 0:
                clean_u.append(u_val)
                clean_n.append(n)
                
        return len(clean_u), clean_n, clean_u

    def get_delta_entropy_theta_category_subset(self, p_idx, nodes, new_val):
        """Evaluates changing a subset of nodes to a single new parameter category value."""
        if not nodes:
            return 0.0
            
        old_ll_sum = sum(self.node_ll[i] for i in nodes)
        new_ll_sum = 0.0
        for i in nodes:
            theta_i_new = self.theta[i, :].copy()
            theta_i_new[p_idx] = new_val
            new_ll_sum += self._calc_node_ll(i, theta_i=theta_i_new)
            
        delta_ll = new_ll_sum - old_ll_sum
        
        new_n_counts = list(self.theta_n[p_idx])
        new_u = list(self.theta_u[p_idx])
        
        for i in nodes:
            old_k = self.node_to_theta_cat[p_idx][i]
            new_n_counts[old_k] -= 1
            
        k_idx = next((k for k, u_val in enumerate(new_u) if abs(u_val - new_val) < 1e-6), None)
        if k_idx is not None:
            new_n_counts[k_idx] += len(nodes)
        else:
            new_u.append(new_val)
            new_n_counts.append(len(nodes))
            
        clean_u = []
        clean_n = []
        for u_val, n in zip(new_u, new_n_counts):
            if n > 0:
                clean_u.append(u_val)
                clean_n.append(n)
                
        new_K = len(clean_u)
        
        old_prior = compute_mdl_theta_prior(self.N, self.theta_K[p_idx], self.theta_n[p_idx], self.theta_u[p_idx])
        new_prior = compute_mdl_theta_prior(self.N, new_K, clean_n, clean_u)
        delta_prior = new_prior - old_prior
        
        return -delta_ll + delta_prior

    def apply_theta_category_update(self, i, p_idx, new_val):
        self.apply_theta_category_update_subset(p_idx, [i], new_val)

    def apply_theta_category_update_subset(self, p_idx, nodes, new_val):
        for i in nodes:
            self.theta[i, p_idx] = new_val
            self.node_ll[i] = self._calc_node_ll(i)
            
            old_k = self.node_to_theta_cat[p_idx][i]
            self.theta_n[p_idx][old_k] -= 1
            
        k_idx = next((k for k, u_val in enumerate(self.theta_u[p_idx]) if abs(u_val - new_val) < 1e-6), None)
        if k_idx is None:
            self.theta_u[p_idx].append(new_val)
            self.theta_n[p_idx].append(0)
            self.theta_K[p_idx] += 1
            k_idx = len(self.theta_u[p_idx]) - 1
            
        for i in nodes:
            self.theta_n[p_idx][k_idx] += 1
            self.node_to_theta_cat[p_idx][i] = k_idx
            
        self._gc_theta_categories(p_idx)

    def _gc_theta_categories(self, p_idx):
        if 0 not in self.theta_n[p_idx]:
            return
            
        old_to_new_k = {}
        clean_u = []
        clean_n = []
        
        new_k = 0
        for old_k, n in enumerate(self.theta_n[p_idx]):
            if n > 0:
                clean_u.append(self.theta_u[p_idx][old_k])
                clean_n.append(n)
                old_to_new_k[old_k] = new_k
                new_k += 1
                
        self.theta_u[p_idx] = clean_u
        self.theta_n[p_idx] = clean_n
        self.theta_K[p_idx] = len(clean_u)
        
        for i in list(self.node_to_theta_cat[p_idx].keys()):
            old_k = self.node_to_theta_cat[p_idx][i]
            if old_k in old_to_new_k:
                self.node_to_theta_cat[p_idx][i] = old_to_new_k[old_k]
