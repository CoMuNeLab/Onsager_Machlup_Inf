import sys

with open("run_grid_search.py", "r") as f:
    content = f.read()
    
# Replace generator import
content = content.replace(
    "from inference_pipeline.data_generation.network_generator import generate_scale_free_W",
    "from inference_pipeline.data_generation.network_generator import generate_erdos_renyi_W"
)

# Replace generator call
content = content.replace(
    "W_true = generate_scale_free_W(N, m, seed=seed_network)",
    "W_true = generate_erdos_renyi_W(N, m=m, seed=seed_network)"
)

# Replace output directory
content = content.replace(
    '"results", "grid_results"',
    '"results", "grid_results_er"'
)

# Replace description
content = content.replace(
    'description="Run GRN Inference Grid Search"',
    'description="Run GRN Inference Grid Search (E-R Topology)"'
)

with open("run_grid_search_er.py", "w") as f:
    f.write(content)

print("Created run_grid_search_er.py")
