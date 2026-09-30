import sys

def patch_file(filename):
    with open(filename, "r") as f:
        content = f.read()

    # Update row dict
    old_row = """            "AUPRC": top_metrics.get("auprc", np.nan),"""
    new_row = """            "AUPRC": top_metrics.get("auprc", np.nan),
            "AUPRC_Ratio": top_metrics.get("auprc_ratio", np.nan),
            "AUPRC_Norm": top_metrics.get("auprc_norm", np.nan),"""
            
    if '"AUPRC_Ratio"' not in content:
        content = content.replace(old_row, new_row)
        
    # Update fieldnames
    old_fields = """        "AUROC", "AUPRC", "MSE", "Pearson", "Spearman", "RBO", "Wasserstein","""
    new_fields = """        "AUROC", "AUPRC", "AUPRC_Ratio", "AUPRC_Norm", "MSE", "Pearson", "Spearman", "RBO", "Wasserstein","""
    
    if '"AUPRC_Ratio"' not in content:
        content = content.replace(old_fields, new_fields)

    with open(filename, "w") as f:
        f.write(content)

patch_file("run_grid_search.py")
patch_file("run_grid_search_er.py")
patch_file("compute_grid_metrics.py")
print("Successfully patched grid search scripts.")

