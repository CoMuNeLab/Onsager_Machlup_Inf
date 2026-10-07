![InferenceBio Project Overview](papers_figures/Fig_Github.png)

# Resolving Structural Uncertainty in Stochastic Dynamical Systems via Ensemble-Based Bayesian Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

Codebase and reproduction workflows for the paper:
> **"Resolving Structural Uncertainty in Stochastic Dynamical Systems via Ensemble-Based Bayesian Inference"**  
> Francesco Zambelli, Vincenzo D'Andrea, Manlio De Domenico.

---

## 📌 Overview
Complex biological and physical systems—from gene regulatory networks (GRNs) to neural circuits—often operate under severe structural uncertainty, observational noise, and finite-sampling regimes. This framework implements a nonparametric Bayesian formulation built on stochastic differential equations (SDEs) and Markov Chain Monte Carlo (MCMC) over random graph priors using `graph-tool`.

By sampling an **ensemble of plausible network topologies and kinetic parameters** rather than relying on single point-estimate reconstructions, this approach enables:
1. **Accurate Structural Recovery:** Disentangling direct interactions from indirect correlations without ad-hoc regularization.
2. **Robust Predictive Simulation:** Unbiased forecasting of out-of-sample steady states and knock-out/perturbation response.
3. **Resilience to Model Mismatch:** Robust inference even when observing macroscopic rate dynamics (e.g., Wilson-Cowan neural populations).

---

## 📂 Repository Structure

```text
├── Main_Pipeline.ipynb               # Interactive end-to-end demo (synthetic GRN simulation & MCMC inference)
├── Grid_Search_Analysis.ipynb        # Comprehensive analysis of scale-free & Erdős-Rényi parameter sweeps
├── Wilson_Cowan_Inference_Benchmark.ipynb # Benchmark on neural population dynamics under model mismatch (Fig. 4)
│
├── gene_expr/                         # High-performance C++ Bayesian MCMC engine (graph-tool extension)
│   ├── GeneExprBlockState.hh
│   ├── GeneExprBlockState.cc
│   └── setup.py
│
├── inference_pipeline/                # Python core framework
│   ├── sim/                          # SDE network dynamics simulation (Ornstein-Uhlenbeck, Euler-Maruyama)
│   ├── inference/                    # MCMC orchestration, topology reconstruction metrics
│   ├── prediction/                   # Out-of-sample simulation & perturbation response metrics
│   └── io/                           # Serialization & binary array I/O
│
├── scripts/
│   ├── benchmarking/                 # Reproduction scripts for paper benchmarks
│   │   ├── run_publication_benchmark.py # SDE vs discrete/Taylor discretization benchmark (Fig. 3c)
│   │   ├── plot_publication_figure.py   # Generates Fig. 3c
│   │   ├── experiment_sparsity.py       # Global vs Local regularization mode benchmark (Fig. 3a)
│   │   ├── replot_sparsity.py           # Generates Fig. 3a
│   │   ├── experiment_overfitting.ipynb # Multi-method competitor benchmark (Fig. 3d)
│   │   └── methods/                     # Benchmark competitors (Sparse ODE, Random Forest, Corr)
│   ├── analysis/                     # Paper analysis & perturbation response scripts
│   │   ├── plot_paper_perturbation.py   # Generates Fig. 3e, f (point-estimate vs ensemble perturbation)
│   │   ├── plot_transient_vs_stationary.py # Generates Fig. 3b (regime comparison across scales)
│   │   ├── reconstruction_metrics.py    # AUROC, AUPRC, SHD, Wasserstein, RBO evaluation
│   │   ├── perturbation_metrics.py      # Response correlation, CI calibration, log-likelihood
│   │   └── tests/                       # Unit tests for evaluation metrics and MCMC engine
│   └── grid_search/                  # Massive scale-free and Erdős-Rényi parameter sweeps
│       ├── run_grid_search.py           # Sweep runner for Scale-Free networks
│       ├── run_grid_search_er.py        # Sweep runner for Erdős-Rényi networks
│       └── compute_grid_metrics.py      # Aggregator for trial metrics into summary CSVs
│
├── results/                          # Curated experimental data & summaries
│   ├── benchmarking/
│   │   ├── discretization/           # Benchmark metrics for SDE continuous vs discrete models
│   │   └── regularization/           # Local vs Global prior mode trial data
│   ├── perturbation/                 # MCMC ensemble samples & ground-truth binary files
│   ├── grid_results/                 # Scale-free parameter grid summary (performance_metrics.csv)
│   └── grid_results_er/              # Erdős-Rényi parameter grid summary (performance_metrics.csv)
│
├── images/                           # Generated figures organized by analysis module
│   ├── benchmarking/
│   ├── analysis/
│   └── grid_search/
│
└── papers_figures/                   # Final publication figures (PDF and high-res PNG)
```

---

## 🚀 Installation & Requirements

The project requires Python $\ge$ 3.10 and **[`graph-tool`](https://graph-tool.skewed.de/)**. The recommended installation is via `conda`:

```bash
# 1. Create and activate conda environment with graph-tool
conda create -n gt -c conda-forge python=3.12 graph-tool scipy numpy pandas matplotlib seaborn scikit-learn
conda activate gt

# 2. Compile the C++ MCMC extension
cd gene_expr
python setup.py build_ext --inplace
cd ..
```

---

## 📊 Reproducing Paper Experiments and Figures

| Figure | Description | Reproduction Script / Notebook |
| :--- | :--- | :--- |
| **Fig. 1** | Conceptual workflow & ensemble posterior representation | Diagram / [`papers_figures/Fig1.pdf`](papers_figures/Fig1.pdf) |
| **Fig. 2** | Description length & MCMC entropy convergence | [`Main_Pipeline.ipynb`](Main_Pipeline.ipynb) |
| **Fig. 3a** | Global vs. Local regularization / sparsity selection | `python scripts/benchmarking/replot_sparsity.py` |
| **Fig. 3b** | Transient vs. Stationary regime comparison across scales | `python scripts/analysis/plot_transient_vs_stationary.py` |
| **Fig. 3c** | SDE Exact vs. Continuous/Discrete likelihood discretization | `python scripts/benchmarking/plot_publication_figure.py` |
| **Fig. 3d** | Competitor benchmark under overfitting & finite data | [`scripts/benchmarking/experiment_overfitting.ipynb`](scripts/benchmarking/experiment_overfitting.ipynb) |
| **Fig. 3e, f** | Out-of-sample perturbation response & ensemble advantage | `python scripts/analysis/plot_paper_perturbation.py` |
| **Fig. 4** | Model mismatch benchmark on Wilson-Cowan rate dynamics | [`Wilson_Cowan_Inference_Benchmark.ipynb`](Wilson_Cowan_Inference_Benchmark.ipynb) |
| **Fig. S1** | Erdős-Rényi vs Scale-Free architecture comparison | [`Grid_Search_Analysis.ipynb`](Grid_Search_Analysis.ipynb) |

---

## 🧪 Running Tests

Verify the installation and core metric pipelines:
```bash
python scripts/analysis/tests/test_evaluation_metrics.py
python scripts/analysis/tests/test_gene_expr_inference.py
```

---

## 📜 Citation
If you use this codebase or method in your research, please cite:
```bibtex
@article{zambelli2026resolving,
  title={Resolving Structural Uncertainty in Stochastic Dynamical Systems via Ensemble-Based Bayesian Inference},
  author={Zambelli, Francesco and D'Andrea, Vincenzo and De Domenico, Manlio},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
