import numpy as np

def bisection_search(eval_func, bounds=(-10.0, 10.0), iters=15):
    a, b = bounds
    best_val = 0.0
    best_dS = np.inf
    eps = 1e-4

    for _ in range(iters):
        c = np.random.uniform(a, b)
        dS_c = eval_func(c)
        if dS_c < best_dS:
            best_dS = dS_c
            best_val = c
            
        dS_ce = eval_func(c + eps)
        
        if dS_ce > dS_c:
            b = c
        else:
            a = c
            
    return best_val, best_dS

def find_best_category_value_edge(state, tgt, src, bounds=(-10.0, 10.0), iters=15):
    return bisection_search(lambda v: state.get_delta_entropy_edge(tgt, src, v), bounds, iters)

def find_best_category_value_edge_subset(state, edges, bounds=(-10.0, 10.0), iters=15):
    return bisection_search(lambda v: state.get_delta_entropy_edge_subset(edges, v), bounds, iters)

def find_best_category_value_theta_subset(state, p_idx, nodes, bounds=(-10.0, 10.0), iters=15):
    return bisection_search(lambda v: state.get_delta_entropy_theta_category_subset(p_idx, nodes, v), bounds, iters)


def propose_merge_move_edge(state):
    if state.K < 2:
        return 0, 0
    
    cats = np.random.choice(state.K, 2, replace=False)
    z_k, z_l = state.z[cats[0]], state.z[cats[1]]
    
    edges_k = [edge for edge, cat in state.edge_to_cat.items() if cat == cats[0]]
    edges_l = [edge for edge, cat in state.edge_to_cat.items() if cat == cats[1]]
    edges = edges_k + edges_l
    
    if not edges:
        return 0, 0
        
    new_val, delta_S = find_best_category_value_edge_subset(state, edges)
    
    if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
        state.apply_edge_update_subset(edges, new_val)
        return 1, 1 
        
    return 1, 0


def propose_split_move_edge(state):
    if state.K < 1:
        return 0, 0
        
    cat = np.random.choice(state.K)
    z_m = state.z[cat]
    
    edges = [edge for edge, cat_idx in state.edge_to_cat.items() if cat_idx == cat]
    if len(edges) < 2:
        return 0, 0
        
    # As requested: exact paper logic (loop until convergence inside the proposal)
    sim_state = state.clone()
    
    # 1. Split into two randomly chosen values
    z_k = np.random.uniform(z_m - 1.0, z_m + 1.0)
    z_l = np.random.uniform(z_m - 1.0, z_m + 1.0)
    
    # Randomly assign initially
    E_k = []
    E_l = []
    for edge in edges:
        if np.random.rand() < 0.5:
            E_k.append(edge)
            sim_state.apply_edge_update_subset([edge], z_k)
        else:
            E_l.append(edge)
            sim_state.apply_edge_update_subset([edge], z_l)
            
    # Iterate until convergence (max 5 to prevent infinite loops)
    for _ in range(5):
        changed = False
        # Repeatedly move edges between categories
        for edge in edges:
            curr_val = sim_state.W[edge[0], edge[1]]
            other_val = z_l if abs(curr_val - z_k) < 1e-6 else z_k
            
            dS = sim_state.get_delta_entropy_edge(edge[0], edge[1], other_val)
            if dS < 0:
                sim_state.apply_edge_update(edge[0], edge[1], other_val)
                changed = True
                
        # Re-optimize values via bisection
        curr_E_k = [edge for edge in edges if abs(sim_state.W[edge[0], edge[1]] - z_k) < 1e-6]
        curr_E_l = [edge for edge in edges if abs(sim_state.W[edge[0], edge[1]] - z_l) < 1e-6]
        
        if curr_E_k:
            z_k, _ = find_best_category_value_edge_subset(sim_state, curr_E_k)
            sim_state.apply_edge_update_subset(curr_E_k, z_k)
        if curr_E_l:
            z_l, _ = find_best_category_value_edge_subset(sim_state, curr_E_l)
            sim_state.apply_edge_update_subset(curr_E_l, z_l)
            
        if not changed:
            break
            
    delta_S = sim_state.get_entropy() - state.get_entropy()
    if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
        # Instead of deep copying back, we just overwrite the reference attributes
        # (X_t, X_next are shared)
        state.W = sim_state.W
        state.E = sim_state.E
        state.K = sim_state.K
        state.z = sim_state.z
        state.m_counts = sim_state.m_counts
        state.edge_to_cat = sim_state.edge_to_cat
        state.node_ll = sim_state.node_ll
        return 1, 1
        
    return 1, 0


def propose_merge_move_theta(state, p_idx):
    if state.theta_K[p_idx] < 2:
        return 0, 0
    
    cats = np.random.choice(state.theta_K[p_idx], 2, replace=False)
    u_k, u_l = state.theta_u[p_idx][cats[0]], state.theta_u[p_idx][cats[1]]
    
    nodes_k = [i for i, cat in state.node_to_theta_cat[p_idx].items() if cat == cats[0]]
    nodes_l = [i for i, cat in state.node_to_theta_cat[p_idx].items() if cat == cats[1]]
    nodes = nodes_k + nodes_l
    
    if not nodes:
        return 0, 0
        
    new_val, delta_S = find_best_category_value_theta_subset(state, p_idx, nodes)
    
    if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
        state.apply_theta_category_update_subset(p_idx, nodes, new_val)
        return 1, 1 
        
    return 1, 0


def propose_split_move_theta(state, p_idx):
    if state.theta_K[p_idx] < 1:
        return 0, 0
        
    cat = np.random.choice(state.theta_K[p_idx])
    u_m = state.theta_u[p_idx][cat]
    
    nodes = [i for i, cat_idx in state.node_to_theta_cat[p_idx].items() if cat_idx == cat]
    if len(nodes) < 2:
        return 0, 0
        
    sim_state = state.clone()
    
    u_k = np.random.uniform(u_m - 1.0, u_m + 1.0)
    u_l = np.random.uniform(u_m - 1.0, u_m + 1.0)
    
    for i in nodes:
        if np.random.rand() < 0.5:
            sim_state.apply_theta_category_update_subset(p_idx, [i], u_k)
        else:
            sim_state.apply_theta_category_update_subset(p_idx, [i], u_l)
            
    for _ in range(5):
        changed = False
        for i in nodes:
            curr_val = sim_state.theta[i, p_idx]
            other_val = u_l if abs(curr_val - u_k) < 1e-6 else u_k
            
            dS = sim_state.get_delta_entropy_theta_category_subset(p_idx, [i], other_val)
            if dS < 0:
                sim_state.apply_theta_category_update(i, p_idx, other_val)
                changed = True
                
        curr_nodes_k = [i for i in nodes if abs(sim_state.theta[i, p_idx] - u_k) < 1e-6]
        curr_nodes_l = [i for i in nodes if abs(sim_state.theta[i, p_idx] - u_l) < 1e-6]
        
        if curr_nodes_k:
            u_k, _ = find_best_category_value_theta_subset(sim_state, p_idx, curr_nodes_k)
            sim_state.apply_theta_category_update_subset(p_idx, curr_nodes_k, u_k)
        if curr_nodes_l:
            u_l, _ = find_best_category_value_theta_subset(sim_state, p_idx, curr_nodes_l)
            sim_state.apply_theta_category_update_subset(p_idx, curr_nodes_l, u_l)
            
        if not changed:
            break
            
    delta_S = sim_state.get_entropy() - state.get_entropy()
    if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
        state.theta = sim_state.theta
        state.theta_K = sim_state.theta_K
        state.theta_u = sim_state.theta_u
        state.theta_n = sim_state.theta_n
        state.node_to_theta_cat = sim_state.node_to_theta_cat
        state.node_ll = sim_state.node_ll
        return 1, 1
        
    return 1, 0


def mcmc_sweep(state, beta=1.0):
    """
    Performs one full MCMC sweep: edge updates (including merge/split) and node param updates (with merge/split).
    """
    accept_edge = 0
    accept_edge_merge = 0
    accept_edge_split = 0
    accept_theta = 0
    
    # Topological Edge updates
    for tgt in range(state.N):
        for src in range(state.N):
            if tgt == src: continue
            
            curr_val = state.W[tgt, src]
            
            # 1. Propose setting to 0 (delete)
            if curr_val != 0.0:
                dS = state.get_delta_entropy_edge(tgt, src, 0.0)
                if dS < 0 or np.random.rand() < np.exp(-beta * dS):
                    state.apply_edge_update(tgt, src, 0.0)
                    accept_edge += 1
                    continue
                    
            # 2. Propose assigning to existing category
            if state.K > 0:
                cat_idx = np.random.choice(state.K)
                prop_val = state.z[cat_idx]
                if prop_val != curr_val:
                    dS = state.get_delta_entropy_edge(tgt, src, prop_val)
                    if dS < 0 or np.random.rand() < np.exp(-beta * dS):
                        state.apply_edge_update(tgt, src, prop_val)
                        accept_edge += 1
                        continue
                        
            # 3. Propose assigning to NEW category (bisection)
            # Only do this sometimes to avoid blowing up search space too quickly
            if np.random.rand() < 0.1:
                prop_val, dS = find_best_category_value_edge(state, tgt, src)
                if dS < 0 or np.random.rand() < np.exp(-beta * dS):
                    state.apply_edge_update(tgt, src, prop_val)
                    accept_edge += 1
                    
    # Edge Merge/Split
    for _ in range(5):
        if np.random.rand() < 0.5:
            _, a = propose_merge_move_edge(state)
            accept_edge_merge += a
        else:
            _, a = propose_split_move_edge(state)
            accept_edge_split += a

    # Theta categorical updates
    for p_idx in range(3):
        for i in range(state.N):
            curr_val = state.theta[i, p_idx]
            
            # 1. Propose assigning to existing category
            if state.theta_K[p_idx] > 0:
                cat_idx = np.random.choice(state.theta_K[p_idx])
                prop_val = state.theta_u[p_idx][cat_idx]
                if prop_val != curr_val:
                    dS = state.get_delta_entropy_theta_category_subset(p_idx, [i], prop_val)
                    if dS < 0 or np.random.rand() < np.exp(-beta * dS):
                        state.apply_theta_category_update(i, p_idx, prop_val)
                        accept_theta += 1
                        continue
                        
            # 2. Propose assigning to NEW category
            if np.random.rand() < 0.1:
                prop_val, dS = find_best_category_value_theta_subset(state, p_idx, [i])
                if dS < 0 or np.random.rand() < np.exp(-beta * dS):
                    state.apply_theta_category_update(i, p_idx, prop_val)
                    accept_theta += 1

        # Theta Merge/Split
        for _ in range(5):
            if np.random.rand() < 0.5:
                propose_merge_move_theta(state, p_idx)
            else:
                propose_split_move_theta(state, p_idx)

    return accept_edge, accept_edge_merge, accept_edge_split, accept_theta
