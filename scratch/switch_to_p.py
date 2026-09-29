import sys

def patch_er_script():
    filename = "run_grid_search_er.py"
    with open(filename, "r") as f:
        content = f.read()
    
    # 1. Update run_single_config signature
    content = content.replace(
        "def run_single_config(dt_meas, M_pairs, t_window, N, m, trial_id):",
        "def run_single_config(dt_meas, M_pairs, t_window, N, p, trial_id):"
    )
    
    # 2. Update config name
    content = content.replace(
        'config_name = f"N_{N}_m_{m}_M_{M_pairs}_dt_{dt_meas}_twnd_{t_window[0]}_{t_window[1]}_trial_{trial_id}"',
        'config_name = f"N_{N}_p_{p}_M_{M_pairs}_dt_{dt_meas}_twnd_{t_window[0]}_{t_window[1]}_trial_{trial_id}"'
    )
    
    # 3. Update generator call
    content = content.replace(
        "W_true = generate_erdos_renyi_W(N, m=m, seed=seed_network)",
        "W_true = generate_erdos_renyi_W(N, p=p, seed=seed_network)"
    )
    
    # 4. Add grid_p to parameter grid
    content = content.replace(
        "grid_m = [1, 2, 3]",
        "grid_p = [0.05, 0.1, 0.2]"
    )
    
    # 5. Update dry run
    content = content.replace(
        "grid_m = [1]",
        "grid_p = [0.1]"
    )
    
    # 6. Update all_configs itertools
    content = content.replace(
        "list(itertools.product(grid_dt_meas, grid_M_pairs, grid_t_window, grid_N, grid_m, grid_trials))",
        "list(itertools.product(grid_dt_meas, grid_M_pairs, grid_t_window, grid_N, grid_p, grid_trials))"
    )
    
    # 7. Update unpacking in loop
    content = content.replace(
        "dt_meas, M_pairs, t_window, N, m, trial_id = config",
        "dt_meas, M_pairs, t_window, N, p, trial_id = config"
    )
    
    # 8. Update print statement
    content = content.replace(
        'print(f"\\n[{idx+1}/{total_configs}] ({pct:.1f}%) Processing N={N}, m={m}, M={M_pairs}, dt={dt_meas}, twnd={t_window}, trial={trial_id} ...")',
        'print(f"\\n[{idx+1}/{total_configs}] ({pct:.1f}%) Processing N={N}, p={p}, M={M_pairs}, dt={dt_meas}, twnd={t_window}, trial={trial_id} ...")'
    )
    
    # 9. Update fieldnames to include p
    content = content.replace(
        '"trial", "dt_meas", "M_pairs", "t_window_start", "t_window_end", "N", "m",',
        '"trial", "dt_meas", "M_pairs", "t_window_start", "t_window_end", "N", "p",'
    )
    
    # 10. Update row dict to include p
    content = content.replace(
        '"N": N,\n            "m": m,',
        '"N": N,\n            "p": p,'
    )
    
    with open(filename, "w") as f:
        f.write(content)


def patch_parsers(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Add p to the params default
    if '"p": None' not in content:
        content = content.replace(
            'params = {"N": None, "m": 1, "M": None, "dt": None, "twnd_start": None, "twnd_end": None, "trial": None}',
            'params = {"N": None, "m": 1, "p": None, "M": None, "dt": None, "twnd_start": None, "twnd_end": None, "trial": None}'
        )
        
    if 'elif parts[i] == "p": params["p"] = float(parts[i+1])' not in content:
        content = content.replace(
            'elif parts[i] == "m": params["m"] = int(parts[i+1])',
            'elif parts[i] == "m": params["m"] = int(parts[i+1])\n                elif parts[i] == "p": params["p"] = float(parts[i+1])'
        )
        
    if 'p = params["p"]' not in content:
        content = content.replace(
            'm = params["m"]',
            'm = params["m"]\n            p = params["p"]'
        )
        
    # We shouldn't modify the W_true = generate_scale_free_W in plot/compute scripts blindly
    # because they are not used for ER right now! Actually compute_grid_metrics uses it for SF.
    # To support both, we'd need compute_grid_metrics to know if it's ER or SF.
    # We can detect ER if p is not None.
    
    with open(filename, "w") as f:
        f.write(content)

patch_er_script()
patch_parsers("plot_grid_heatmaps.py")
patch_parsers("compute_grid_metrics.py")
print("Successfully switched run_grid_search_er.py to sweep over p instead of m")
