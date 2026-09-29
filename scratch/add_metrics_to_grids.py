import sys

def patch_script(filename):
    with open(filename, "r") as f:
        content = f.read()

    # 1. Add import
    import_str = "from analysis.reconstruction_metrics import evaluate_topology, evaluate_weights"
    if import_str not in content:
        content = content.replace(
            "from inference_pipeline.inference.metrics import compute_reconstruction_metrics",
            "from inference_pipeline.inference.metrics import compute_reconstruction_metrics\n" + import_str
        )

    # 2. Update metric computation
    old_compute = "metrics = compute_reconstruction_metrics(W_true, W_ensemble[-1])"
    new_compute = """metrics = compute_reconstruction_metrics(W_true, W_ensemble[-1])
        top_metrics = evaluate_topology(W_true, W_ensemble, threshold=0.0, weighted=False)
        weight_metrics = evaluate_weights(W_true, W_ensemble, threshold=0.0)"""
    if old_compute in content and "evaluate_topology" not in content.split(old_compute)[1]:
        content = content.replace(old_compute, new_compute)

    # 3. Update row dictionary
    old_row = """            **metrics,
            "status": "success","""
    new_row = """            **metrics,
            "AUROC": top_metrics.get("auroc", np.nan),
            "AUPRC": top_metrics.get("auprc", np.nan),
            "MSE": weight_metrics.get("mse", np.nan),
            "Pearson": weight_metrics.get("pearson", np.nan),
            "Spearman": weight_metrics.get("spearman", np.nan),
            "RBO": weight_metrics.get("rbo", np.nan),
            "Wasserstein": weight_metrics.get("wasserstein", np.nan),
            "status": "success","""
    if '"AUROC"' not in content:
        content = content.replace(old_row, new_row)

    # 4. Update fieldnames
    old_fields = """        "TP", "FP", "FN", "TN", "TPR (Recall)", "FPR", "Precision", "F1-Score", "SHD",
        "status", "error"
    ]"""
    new_fields = """        "TP", "FP", "FN", "TN", "TPR (Recall)", "FPR", "Precision", "F1-Score", "SHD",
        "AUROC", "AUPRC", "MSE", "Pearson", "Spearman", "RBO", "Wasserstein",
        "status", "error"
    ]"""
    if '"AUROC", "AUPRC"' not in content:
        content = content.replace(old_fields, new_fields)

    with open(filename, "w") as f:
        f.write(content)

patch_script("run_grid_search.py")
patch_script("run_grid_search_er.py")
print("Successfully patched grid search scripts to save new metrics directly.")
