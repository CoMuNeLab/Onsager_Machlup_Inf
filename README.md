![CancerEvo Project Overview](images/Fig_Github.png)

# InferenceBio: Gene Regulatory Network Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
InferenceBio is a computational framework dedicated to the reverse-engineering and topological inference of Gene Regulatory Networks (GRNs) from gene expression time-series data. 

The core of the project relies on Bayesian inference via Markov Chain Monte Carlo (MCMC). By modeling the dynamics of gene expression as a system of Stochastic Differential Equations (SDEs), the framework samples the posterior distribution of potential network topologies (using the `graph-tool` library). It handles both transient (post-perturbation) and stationary regimes, allowing for rigorous analysis of how measurement intervals ($dt_{meas}$), system size ($N$), and the number of independent samples ($M_{pairs}$) impact the structural recovery (F1-Score, SHD) of complex, scale-free networks.

## Folder Structure

- **`analysis/`**: Contains Python scripts dedicated to plotting and analyzing inference results. This includes tools for evaluating node-wise inference, plotting MCMC trajectories, and generating publication-ready figures.
- **`data/`**: The central repository for datasets. This includes ground-truth biological benchmarks (e.g., DREAM4 challenge data) and generated synthetic datasets used for validating the inference pipeline.
- **`entropy_exp/`**: Dedicated to experiments exploring the entropy landscape, description lengths, and potential phase transitions encountered during MCMC traversal of the network topologies.
- **`gene_expr/`**: Contains the core `graph-tool` extension (`GeneExprBlockState`). This module implements the likelihood models and the stochastic differential equation physics governing the gene expression dynamics.
- **`grid_results/`**: Output directory for massive hyperparameter sweeps. Contains `performance_metrics.csv` and individual `.npz` binary results for varying combinations of $N$, $M_{pairs}$, $dt_{meas}$, and sampling regimes.
- **`grn_sim/`**: Modules responsible for the forward problem: generating realistic, scale-free ground-truth topologies and simulating their corresponding gene expression time-series data with noise.
- **`images/`**: The centralized directory for all generated visual assets. This includes performance heatmaps, FacetGrids, network topology graphs, and analysis plots.
- **`inference_pipeline/`**: The core structural pipeline for running Bayesian inference. It includes submodules for data generation, the MCMC engine (burn-in and posterior sampling), evaluation metrics, and export utilities for cross-language compatibility.
- **`inference_python/` & `like_minimization/`**: Alternative or legacy inference modules exploring continuous optimization techniques (e.g., direct likelihood minimization) as opposed to discrete MCMC sampling.
- **`reports/`**: Documentation, literature references, and generated summaries regarding the project's methodologies.
- **`results/`**: General output folder for single-run or custom inferences, storing the resulting `.npz` files (containing ensemble topologies, edge probabilities, and parameter distributions).

## Key Executables & Notebooks
- **`run_grid_search.py`**: Orchestrates a massive parameter sweep across different network conditions. Features built-in checkpointing so it can be cleanly interrupted and resumed.
- **`Grid_Search_Analysis.ipynb`**: A Jupyter Notebook designed to read `grid_results/performance_metrics.csv` and dynamically generate detailed `seaborn` FacetGrid plots to visualize the algorithmic performance.
- **`Main_Pipeline.ipynb`**: An interactive notebook demonstrating the end-to-end inference process for a single network configuration, from data generation to Bayesian posterior sampling.

## Dependencies & Usage
The project relies heavily on `graph-tool` for efficient network manipulation and state modeling.

1. **Environment:** Ensure your conda environment with `graph-tool` is active (`conda activate gt`).
2. **Grid Search:** To launch or resume the grid search pipeline:
   ```bash
   python run_grid_search.py
   ```
3. **Analysis:** Open `Grid_Search_Analysis.ipynb` in Jupyter to visualize the latest results.
