# Gene Expression GRN Inference Extension (`gene_expr`) for Graph-Tool

This package extends Tiago de Paula Peixoto's `graph-tool` Bayesian network inference library (`graph_tool.inference`) with a continuous-time Gene Regulatory Network (GRN) likelihood formulation.

## 1. Theoretical Model & Likelihood Formulation

As described in `model.md`, `PhysRevX.15.011065.pdf`, and `inference_article__method_breakdown.md`, we model the time evolution of expression levels of $N$ genes with the Euler-Maruyama stochastic differential equation (SDE):

$$\frac{dx_i(t)}{dt} = -\gamma_i x_i(t) + V_i \cdot \sigma\left(\sum_j W_{i,j} x_j(t)\right) + \sqrt{2D_i}\, \xi_i(t)$$

where:
- $x_i(t)$: expression level of the $i$-th gene
- $\gamma_i$: decay rate
- $V_i$: maximum expression level
- $W_{i,j}$: interaction strength (regulatory weight from gene $j$ to gene $i$)
- $D_i$: diffusion coefficient
- $\sigma(m) = \frac{1}{1 + e^{-m}}$: non-linear sigmoid activation function

### Onsager-Machlup Conditional Likelihood
For discrete time measurements $t_0, t_1, \dots, t_M$ with interval $\Delta t = t_{k+1} - t_k$, the conditional log-probability density of transition $x_k \to x_{k+1}$ under the Euler-Maruyama scheme is Gaussian:

$$\log P(x_{k+1} | x_k, \Theta) = \mathcal{N}_{\text{lpdf}}\left(x_{k+1} \,\Big|\, \mu_i(x_k), \sigma_i^2\right)$$

with mean and standard deviation:
$$\mu_i(x_k) = x_{i,k} + \Delta t \left(-\gamma_i x_{i,k} + V_i \sigma\left(\sum_j W_{i,j} x_{j,k}\right)\right)$$
$$s_i = \sqrt{D_i \Delta t}$$

To enforce positivity during unrestricted MCMC parameter proposals, the learnable node parameters $\Theta_i$ are parameterized in log-space:
- `theta[0]` $= \log \gamma_i$
- `theta[1]` $= \log V_i$
- `theta[2]` $= \log D_i$

## 2. Package Structure

- **`gene_expr_ext.hh`**: C++ header defining `graph_tool::GeneExprState` which inherits from `ContinuousStateBase<GeneExprState, false, true>` and implements the inline `log_P` transition probability.
- **`gene_expr_ext.cc`**: C++ compilation unit exporting `make_gene_expr_state` and registering `std::shared_ptr<GeneExprState>` via Boost.Python.
- **`Makefile`**: Configured to build `libgene_expr_ext.so` linking against Conda's `libboost_python312.so`, `libstdc++.so.6`, `libgraph_tool_core.so`, and `libgraph_tool_inference.so`.
- **`__init__.py`**: Python wrapper defining `GeneExprBlockState(DynamicsBlockStateBase)`, allowing direct integration with graph-tool's SBM reconstruction and MCMC sweeps.

## 3. Building & Usage

### Compilation
Run `make` inside the `gene_expr` directory:
```bash
cd gene_expr
make
```

### Python Usage Example
```python
import numpy as np
import graph_tool.all as gt
from gene_expr import GeneExprBlockState

# 1. Load or simulate continuous time-series expression data s of shape (N, M)
#    s[i, k] is the expression of gene i at time index k
s = np.load("expression_time_series.npy")

# 2. Instantiate the network reconstruction Bayesian block state
state = GeneExprBlockState(s, dt=0.1)

# 3. Compute initial posterior description length (entropy)
print("Initial entropy:", state.entropy())

# 4. Perform MCMC sweeps over network topology (SBM blocks & edges) and dynamical parameters
for step in range(100):
    dS, nmoves = state.mcmc_sweep(niter=10)
    print(f"Sweep {step+1}: S = {state.entropy():.4f}, accepted moves = {nmoves}")
```

### Running Verification Test
A complete synthetic verification script is provided at the repository root:
```bash
python test_gene_expr_inference.py
```
