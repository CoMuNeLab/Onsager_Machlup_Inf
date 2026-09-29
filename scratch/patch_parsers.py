import sys

robust_parser = """        parts = dirname.split("_")
        params = {"N": None, "m": 1, "M": None, "dt": None, "twnd_start": None, "twnd_end": None, "trial": None}
        
        try:
            for i in range(len(parts)):
                if parts[i] == "N": params["N"] = int(parts[i+1])
                elif parts[i] == "m": params["m"] = int(parts[i+1])
                elif parts[i] == "M": params["M"] = int(parts[i+1])
                elif parts[i] == "dt": params["dt"] = float(parts[i+1])
                elif parts[i] == "twnd": 
                    params["twnd_start"] = float(parts[i+1])
                    params["twnd_end"] = float(parts[i+2])
                elif parts[i] == "trial": params["trial"] = int(parts[i+1])
                
            N = params["N"]
            m = params["m"]
            M = params["M"]
            dt = params["dt"]
            twnd_start = params["twnd_start"]
            twnd_end = params["twnd_end"]
            trial = params["trial"]
            
            if N is None or M is None or dt is None or twnd_start is None:
                continue
        except (IndexError, ValueError):
            continue"""

def patch_file(filename):
    with open(filename, "r") as f:
        content = f.read()
        
    old_parser = """        parts = dirname.split("_")
        try:
            N = int(parts[1])
            M = int(parts[3])
            dt = float(parts[5])
            twnd_start = float(parts[7])
            twnd_end = float(parts[8])
            trial = int(parts[10])
        except (IndexError, ValueError):
            continue"""
            
    content = content.replace(old_parser, robust_parser)
    
    # In compute_grid_metrics.py we need to handle `m` in generating W_true
    # W_true = generate_scale_free_W(N, m=1, seed=seed_network)
    content = content.replace("W_true = generate_scale_free_W(N, m=1, seed=seed_network)", "W_true = generate_scale_free_W(N, m=m, seed=seed_network)")
    
    # Also add "m" to records.append in both files.
    # We do a naive replacement to add "m": m to the dictionary
    content = content.replace('"N": N,', '"N": N,\n            "m": m,')
    
    # In plot_grid_heatmaps.py we also need to groupby 'm'
    content = content.replace('df.groupby(["N", "M_pairs", "dt_meas", "Regime"])', 'df.groupby(["N", "m", "M_pairs", "dt_meas", "Regime"])')
    
    # In plot_grid_heatmaps.py, we iterate over unique N, Regime. Now also over m?
    # We should just let it be, but let's change the loop to also iterate over m if present.
    #    for n_val in df_agg["N"].unique():
    #       for reg in df_agg["Regime"].unique():
    content = content.replace('for n_val in df_agg["N"].unique():', 'for n_val in df_agg["N"].unique():\n        for m_val in df_agg["m"].unique():')
    content = content.replace('subset = df_agg[(df_agg["N"] == n_val) & (df_agg["Regime"] == reg)]', 'subset = df_agg[(df_agg["N"] == n_val) & (df_agg["m"] == m_val) & (df_agg["Regime"] == reg)]')
    content = content.replace('f"AUPRC (N={n_val}, {reg} Regime)"', 'f"AUPRC (N={n_val}, m={m_val}, {reg} Regime)"')
    content = content.replace('f"heatmap_AUPRC_N{n_val}_{reg}.png"', 'f"heatmap_AUPRC_N{n_val}_m{m_val}_{reg}.png"')
    content = content.replace('f"AUROC (N={n_val}, {reg} Regime)"', 'f"AUROC (N={n_val}, m={m_val}, {reg} Regime)"')
    content = content.replace('f"heatmap_AUROC_N{n_val}_{reg}.png"', 'f"heatmap_AUROC_N{n_val}_m{m_val}_{reg}.png"')
    content = content.replace('f"Weighted MAE (N={n_val}, {reg} Regime)"', 'f"Weighted MAE (N={n_val}, m={m_val}, {reg} Regime)"')
    content = content.replace('f"heatmap_MAE_N{n_val}_{reg}.png"', 'f"heatmap_MAE_N{n_val}_m{m_val}_{reg}.png"')
    
    with open(filename, "w") as f:
        f.write(content)

patch_file("plot_grid_heatmaps.py")
patch_file("compute_grid_metrics.py")
print("Patched parsers successfully.")
