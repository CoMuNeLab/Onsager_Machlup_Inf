import sys

def update_script(filename, is_er=False):
    with open(filename, "r") as f:
        content = f.read()
    
    # 1. Update run_single_config signature
    content = content.replace(
        "def run_single_config(dt_meas, M_pairs, t_window, N, trial_id):",
        "def run_single_config(dt_meas, M_pairs, t_window, N, m, trial_id):"
    )
    
    # 2. Remove fixed m
    content = content.replace("        m = 1\n", "")
    
    # 3. Update config name
    content = content.replace(
        'config_name = f"N_{N}_M_{M_pairs}_dt_{dt_meas}_twnd_{t_window[0]}_{t_window[1]}_trial_{trial_id}"',
        'config_name = f"N_{N}_m_{m}_M_{M_pairs}_dt_{dt_meas}_twnd_{t_window[0]}_{t_window[1]}_trial_{trial_id}"'
    )
    
    # 4. Add grid_m to parameter grid
    content = content.replace(
        "grid_N = [10, 20, 50, 100]",
        "grid_N = [10, 20, 50, 100]\n    grid_m = [1, 2, 3]"
    )
    
    # 5. Update dry run
    content = content.replace(
        "grid_N = [10, 20]",
        "grid_N = [10, 20]\n        grid_m = [1]"
    )
    
    # 6. Update all_configs itertools
    content = content.replace(
        "list(itertools.product(grid_dt_meas, grid_M_pairs, grid_t_window, grid_N, grid_trials))",
        "list(itertools.product(grid_dt_meas, grid_M_pairs, grid_t_window, grid_N, grid_m, grid_trials))"
    )
    
    # 7. Update unpacking in loop
    content = content.replace(
        "dt_meas, M_pairs, t_window, N, trial_id = config",
        "dt_meas, M_pairs, t_window, N, m, trial_id = config"
    )
    
    # 8. Update print statement
    content = content.replace(
        'print(f"\\n[{idx+1}/{total_configs}] ({pct:.1f}%) Processing N={N}, M={M_pairs}, dt={dt_meas}, twnd={t_window}, trial={trial_id} ...")',
        'print(f"\\n[{idx+1}/{total_configs}] ({pct:.1f}%) Processing N={N}, m={m}, M={M_pairs}, dt={dt_meas}, twnd={t_window}, trial={trial_id} ...")'
    )
    
    # 9. Update fieldnames to include m
    content = content.replace(
        '"trial", "dt_meas", "M_pairs", "t_window_start", "t_window_end", "N",',
        '"trial", "dt_meas", "M_pairs", "t_window_start", "t_window_end", "N", "m",'
    )
    
    # 10. Update row dict to include m
    content = content.replace(
        '"N": N,',
        '"N": N,\n            "m": m,'
    )
    
    with open(filename, "w") as f:
        f.write(content)

update_script("run_grid_search.py")
update_script("run_grid_search_er.py")
print("Updated grid search scripts.")

