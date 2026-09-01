import numpy as np

def find_best_category_value(state, tgt, src, bounds=(-10.0, 10.0), iters=15):
    """
    Finds a good continuous value for a new weight category for edge (tgt, src).
    Uses a random bisection search as described in the paper to find the global 
    minimum of the potentially multimodal objective function.
    """
    a, b = bounds
    best_val = 0.0
    best_dS = np.inf
    eps = 1e-4
    
    for _ in range(iters):
        # Sample midpoint uniformly at random
        c = np.random.uniform(a, b)
        
        if abs(c) < 1e-9:
            c = 1e-5 # Avoid exactly zero
            
        dS_c = state.get_delta_entropy_edge(tgt, src, c)
        
        if dS_c < best_dS:
            best_dS = dS_c
            best_val = c
            
        # Estimate local gradient to determine which half to discard
        dS_c_plus = state.get_delta_entropy_edge(tgt, src, c + eps)
        
        if dS_c_plus > dS_c:
            # Objective is increasing, minimum should be to the left
            b = c
        else:
            # Objective is decreasing, minimum should be to the right
            a = c
            
    return best_val, best_dS

def mcmc_sweep(state, beta=1.0, niter=1, weight_bounds=(-10.0, 10.0), 
               theta_bounds=[(-5.0, 3.0), (-3.0, 4.0), (-8.0, 1.0)], 
               theta_step=0.1):
    """
    Performs an MCMC sweep over the network topology AND node parameters (theta).
    
    Parameters:
    - state: GRNInferenceState object
    - beta: Inverse temperature for simulated annealing
    - niter: Number of full passes over the edges and nodes
    - weight_bounds: Tuple (min, max) for continuous edge weights
    - theta_bounds: List of (min, max) for [log(gamma), log(V), log(D)]
    - theta_step: Standard deviation for Gaussian proposal jumps of theta
    
    Returns:
    - dS_total: Total change in entropy
    - n_moves: Number of accepted topological moves
    - n_theta_moves: Number of accepted parameter moves
    """
    dS_total = 0.0
    n_moves = 0
    n_theta_moves = 0
    
    N = state.N
    
    for _ in range(niter):
        # 1. Topological Search (Edges)
        edges = [(i, j) for i in range(N) for j in range(N) if i != j]
        np.random.shuffle(edges)
        
        for tgt, src in edges:
            current_val = state.W[tgt, src]
            
            # Propose moves
            proposals = []
            
            # 1a. Propose deletion
            if current_val != 0.0:
                dS = state.get_delta_entropy_edge(tgt, src, 0.0)
                proposals.append((0.0, dS))
                
            # 1b. Propose assigning to an existing category
            for z_val in state.z:
                if z_val != current_val:
                    dS = state.get_delta_entropy_edge(tgt, src, z_val)
                    proposals.append((z_val, dS))
                    
            # 1c. Propose creating a new category
            new_val, new_dS = find_best_category_value(state, tgt, src, bounds=weight_bounds)
            if new_val not in state.z and new_val != current_val:
                proposals.append((new_val, new_dS))
                
            if not proposals:
                continue
                
            prop_val, prop_dS = proposals[np.random.randint(len(proposals))]
            
            # Acceptance probability
            if prop_dS < 0:
                accept = True
            else:
                p_accept = np.exp(-beta * prop_dS)
                accept = (np.random.rand() < p_accept)
                
            if accept:
                state.apply_edge_update(tgt, src, prop_val)
                dS_total += prop_dS
                n_moves += 1

        # 2. Parameter Search (Theta)
        nodes = list(range(N))
        np.random.shuffle(nodes)
        
        for i in nodes:
            current_theta = state.theta[i].copy()
            # Propose a Gaussian random walk jump
            prop_theta = current_theta + np.random.normal(0, theta_step, size=3)
            
            # Check bounds (uniform prior)
            in_bounds = True
            for k in range(3):
                if not (theta_bounds[k][0] <= prop_theta[k] <= theta_bounds[k][1]):
                    in_bounds = False
                    break
            
            if not in_bounds:
                continue
                
            prop_dS = state.get_delta_entropy_theta(i, prop_theta)
            
            # Acceptance probability
            if prop_dS < 0:
                accept = True
            else:
                p_accept = np.exp(-beta * prop_dS)
                accept = (np.random.rand() < p_accept)
                
            if accept:
                state.apply_theta_update(i, prop_theta)
                dS_total += prop_dS
                n_theta_moves += 1
                
    return dS_total, n_moves, n_theta_moves
