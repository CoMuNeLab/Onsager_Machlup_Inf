# Technical Guide: Extending Graph-Tool with Custom Continuous Dynamical Likelihoods

This document provides a comprehensive technical breakdown of how the continuous-time Gene Regulatory Network (GRN) Bayesian inference extension (`gene_expr`) was created for Tiago de Paula Peixoto's `graph-tool` library (`graph_tool.inference`). 

It details the architecture of `graph-tool`'s inference engine, lists and explains the most critical files in the `graph-tool` source code to reference when building custom extensions, and explains the engineering and mathematical decisions behind our C++/Python implementation.

---

## 1. Graph-Tool Inference Architecture & Essential Source Files

`graph-tool` uses a hybrid C++/Python architecture for network reconstruction and dynamical inference:
- **C++ Template-Dispatched Core**: Performs computationally intensive likelihood calculations, time-series iteration, and MCMC delta-entropy evaluations across graph edges and nodes.
- **Python High-Level Interface**: Manages Stochastic Block Model (SBM) priors, MCMC sweeps, parameter proposal distributions, and user-facing state containers.

When extending `graph-tool` with a new dynamical likelihood model (such as an ODE/SDE system), you interact with the following core source files in the `graph-tool` repository:

### A. Core C++ Inference Headers (`/include/inference/`)

1. **`inference/uncertain/dynamics_continuous.hh`**
   - **Role**: The primary header defining continuous-time dynamical state models (such as `LVState`, `LinearNormalState`, and `NormalGlauberState`).
   - **Key Class**: `ContinuousStateBase<State, is_discrete, is_continuous>`
   - **How it works**: Your custom state class must inherit from `ContinuousStateBase<YourState, false, true>` and implement a high-performance, inline conditional log-likelihood method:
     ```cpp
     [[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
     double log_P(const std::vector<double>& theta, double m, double x, double nx);
     ```
     - `theta`: Vector of learnable node-level parameters for vertex $i$.
     - `m`: Weighted sum of incoming inputs from neighbors: $m_i = \sum_j W_{i,j} x_j(t)$.
     - `x`: Current observed state value $x_i(t_k)$.
     - `nx`: Next observed state value $x_i(t_{k+1})$.
   - **Why it matters**: `ContinuousStateBase` handles all boilerplate regarding time-series slicing, active-node filtering, and likelihood accumulation across time steps.

2. **`inference/uncertain/dynamics_base.hh` & `inference/uncertain/dynamics_base_imp.hh`**
   - **Role**: Implements `DStateBase`, the polymorphic abstract interface that wraps template-dispatched states.
   - **Key Mechanism**: Contains the outer loops that iterate over the observed time series (`ot`), node states (`os`), and network adjacency/weight matrices to compute the total description length (negative log-posterior):
     $$\Sigma = -\log P(X \mid \Theta, G) - \log P(\Theta, G)$$
   - **Why it matters**: During MCMC sweeps, when an edge addition/deletion or parameter change is proposed, `DStateBase` computes the exact entropy difference $\Delta \Sigma$ by calling your `log_P` method on affected nodes.

3. **`inference/uncertain/dynamics_imp.hh`**
   - **Role**: Defines the factory and Boost.Python registration macros (`MAKE_STATE`, `GEN_DISPATCH`) used to bridge C++ state objects to Python.
   - **Key Mechanism**: Uses template metaprogramming (`block_state::dispatch`) to unpack whatever underlying Stochastic Block Model (`BlockState`) is currently describing the graph topology $G$, wrapping your custom dynamical state in a `std::shared_ptr<DStateBase>` exposed to Python.

4. **`inference/blockmodel/graph_blockmodel.hh`**
   - **Role**: Implements `BlockState` and its specialized variants (degree-corrected, overlapping, layered).
   - **Why it matters**: Network reconstruction in `graph-tool` is formulated as joint Bayesian inference over the network topology $G$ (modeled via an SBM) and the dynamical parameters $\Theta$. This header defines the prior distributions over edges and block partitions.

---

### B. Core Python Inference Modules (`graph_tool/inference/`)

1. **`graph_tool/inference/reconstruction.py`**
   - **Role**: Implements `DynamicsBlockStateBase`, the Python base class for all network reconstruction models (e.g., `LVBlockState`, `IsingGlauberBlockState`).
   - **Key Responsibilities**:
     - Initializes the learnable parameter vector `theta` and its allowed proposal range `theta_range` for each node.
     - Manages MCMC sweeps (`mcmc_sweep(niter=...)`) over both network structure (edges/blocks) and dynamical parameters (`theta`).
     - Invokes the C++ extension factory via `self._dstate = self._make_state()`.
   - **Why it matters**: Your custom Python wrapper subclass must inherit from `DynamicsBlockStateBase` and override `_make_state()` to connect to your compiled C++ extension.

2. **`graph_tool/inference/uncertain/dynamics.py`**
   - **Role**: Provides forward-simulation and inspection classes (`DynamicsStateBase`) for simulating trajectories once a network and parameters have been inferred.

---

## 2. Step-by-Step Technical Implementation of `gene_expr`

### Step 1: Mathematical Translation into `GeneExprState` (`gene_expr/gene_expr_ext.hh`)

We modeled gene expression dynamics using the Euler-Maruyama SDE:
$$\frac{dx_i(t)}{dt} = -\gamma_i x_i(t) + V_i \sigma\left(\sum_j W_{i,j} x_j(t)\right) + \sqrt{2D_i}\, \xi_i(t)$$
where $\sigma(m) = \frac{1}{1 + e^{-m}}$.

In `gene_expr_ext.hh`, we implemented `GeneExprState` inheriting from `ContinuousStateBase<GeneExprState, false, true>`. 

#### Critical Design Decision: Log-Space Parameterization
In `graph-tool`'s `DynamicsBlockStateBase`, node parameters in `theta` are updated during MCMC sweeps using unrestricted Gaussian proposals in $(-\infty, \infty)$. If physical parameters like decay rate $\gamma_i$, maximum expression $V_i$, and diffusion coefficient $D_i$ were stored directly in `theta`, MCMC proposals could generate negative values, leading to unphysical drift or fatal math errors (`sqrt` of negative diffusion, `log(0)`).

To guarantee unconditional positivity without modifying `graph-tool`'s internal proposal engine, we parameterized `theta` in **log-space**:
- `theta[0]` $= \log \gamma_i \implies \gamma_i = \exp(\theta_0)$
- `theta[1]` $= \log V_i \implies V_i = \exp(\theta_1)$
- `theta[2]` $= \log D_i \implies D_i = \exp(\theta_2)$

#### Likelihood Calculation (`log_P`)
For discrete time step $\Delta t$, the Euler-Maruyama transition probability density is Gaussian:
$$P(x_{k+1} \mid x_k, \Theta) = \mathcal{N}\left(x_{k+1} \;\Big|\; \mu = x_k + \Delta t \left(-\gamma_i x_k + V_i \sigma(m)\right), \; \sigma^2 = D_i \Delta t\right)$$

In our C++ implementation:
```cpp
[[gnu::pure]] [[gnu::flatten]] [[gnu::always_inline]]
double log_P(const std::vector<double>& theta, double m, double x, double nx)
{
    double gamma = exp(theta[0]);
    double V = exp(theta[1]);
    double D = exp(theta[2]);

    double sig_m = 1.0 / (1.0 + exp(-m));
    double mu = x + _dt * (-gamma * x + V * sig_m);
    
    double s = sqrt(D * _dt);
    // Optimization: log(s) = 0.5 * (log(D) + log(dt)) = 0.5 * (theta[2] + _ldt)
    double log_s = 0.5 * (theta[2] + _ldt);

    return norm_lpdf(nx, mu, s, log_s);
}
```

---

### Step 2: C++ / Python Binding & Registration (`gene_expr/gene_expr_ext.cc`)

Creating a standalone shared C++ extension (`.so`) for `graph-tool` requires careful handling of Boost.Python symbol tables and type converters.

#### Resolving Module Registry Crashes
In `graph-tool`'s internal C++ code, states are registered using an internal `module_registry.hh` mechanism (`EvokeRegistry`), which uses static global initialization hooks. **Attempting to use `module_registry.hh` inside a separate out-of-tree shared object causes segmentation faults** at Python import time due to symbol collisions with `libgraph_tool_inference.so`'s existing registry.

**Solution**: We bypassed `module_registry.hh` completely and used standard Boost.Python module definition:
```cpp
BOOST_PYTHON_MODULE(libgene_expr_ext)
{
    using namespace boost::python;
    def("make_gene_expr_state", &make_gene_expr_state);
    class_<DState, bases<DStateBase>, std::shared_ptr<DState>,
           boost::noncopyable>("GeneExprState", no_init);
}
```

#### Resolving `std::shared_ptr` Converter Errors
When Python calls `make_gene_expr_state`, our C++ factory returns a `python::object` wrapping a `std::shared_ptr<GeneExprState>`. Without explicit class registration, Python throws:
`TypeError: No to_python (by-value) converter found for C++ type: std::shared_ptr<graph_tool::GeneExprState>`.

Adding the `class_<DState, bases<DStateBase>, std::shared_ptr<DState>, ...>("GeneExprState", no_init)` registration inside `BOOST_PYTHON_MODULE` explicitly registers the inheritance hierarchy and shared-pointer conversion rules with Boost.Python's runtime type system.

---

### Step 3: Compilation & ABI Matching (`gene_expr/Makefile`)

To prevent ABI incompatibilities (which manifest as obscure crashes, symbol lookup errors, or memory corruption), our `Makefile` enforces strict alignment with `graph-tool`'s build environment:
1. **C++ Standard**: Must use `-std=gnu++17` (matching `graph-tool`'s C++17 requirement and Boost 1.80+ headers).
2. **Include Paths**: Includes headers from Conda's `graph_tool/include/`, `boost-workaround/`, `pcg-cpp/`, and `numpy/_core/include/`.
3. **Dynamic Library Linking & RPATH**:
   ```makefile
   LDFLAGS = -shared \
       -L$(CONDA_PREFIX)/lib \
       -L$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool \
       -L$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/inference \
       -Wl,-rpath,$(CONDA_PREFIX)/lib:$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool:$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/inference \
       -lboost_python312 -lboost_context -lgraph_tool_core -lgraph_tool_inference
   ```
   - `-Wl,-rpath,...` ensures that when Python loads `libgene_expr_ext.so`, the Linux dynamic linker explicitly resolves symbols against the Conda environment's `libstdc++.so.6` and `libboost_python312.so` instead of system libraries.

---

### Step 4: Python Reconstruction Wrapper (`gene_expr/__init__.py`)

To make the C++ extension seamless for users, we implemented `GeneExprBlockState` inheriting from `DynamicsBlockStateBase`:
```python
class GeneExprBlockState(DynamicsBlockStateBase):
    def __init__(self, s, g=None, self_loops=True, dt=1.0, **kwargs):
        # Default initialization: 3 node parameters per vertex (log(gamma), log(V), log(D))
        q = 3
        kwargs.setdefault("theta", [np.log(0.5), np.log(2.0), np.log(0.05)])
        kwargs.setdefault("theta_range", [(-5.0, 3.0), (-3.0, 4.0), (-8.0, 1.0)])
        kwargs.setdefault("tdelta", [1e-8] * q)
        kwargs.setdefault("tdelta_min", [1e-8] * q)
        kwargs.setdefault("disable_tdist", True)

        DynamicsBlockStateBase.__init__(
            self, s, g=g, self_loops=self_loops, q_t=q,
            **dict(dict(directed=True, dt=dt), **kwargs)
        )
        self.theta_first = False

    def _make_state(self):
        # Bridge to our compiled C++ extension factory
        return libgene_expr_ext.make_gene_expr_state(
            self._state, self.ot, self.os, self.oactive, self.params
        )
```
- **Why `q_t = 3` and `theta` initial values**: Informs `DynamicsBlockStateBase` that each node requires 3 learnable parameters (`log(gamma)`, `log(V)`, `log(D)`).
- **Why `tdelta = [1e-8] * q` and `tdelta_min = [1e-8] * q` are CRITICAL**: By default, `DynamicsBlockStateBase.__init__` initializes `tdelta` and `tdelta_min` as single-element lists (`[1e-8]`), which are passed to the C++ member vector `std::vector<double> _tdelta`. When `q_t > 1` (as in our 3-parameter GRN model), any evaluation of the Laplace/L1 prior (`tl1`) or parameter bisection in C++ (`dynamics_util.hh::qlaplace_lprob`) accesses `_tdelta[1]` and `_tdelta[2]`. If `_tdelta` was not explicitly dimensioned to `q_t=3` in Python, C++ performs an out-of-bounds read on `_tdelta`, retrieving uninitialized memory or `0.0`. This causes `qlaplace_lprob(x, tl1[l], _tdelta[l], false)` to evaluate to `-inf`, leading to `state.entropy()` returning unphysical `inf` values. Explicitly providing `tdelta=[1e-8]*3` and `tdelta_min=[1e-8]*3` prevents out-of-bounds reads and ensures numerically stable, linear entropy progression across all time series lengths.

---

## 3. Summary of Workflow for Extending Graph-Tool

1. **Header Definition**: Create `your_model.hh` inheriting from `ContinuousStateBase` (or `DiscreteStateBase`) and implement inline `log_P()`.
2. **Parameterization**: Ensure all parameters in `theta` map cleanly over $(-\infty, \infty)$ using transformations (e.g., `exp()`, `sigmoid()`).
3. **C++ Module Export**: Create `your_model.cc` using `BOOST_PYTHON_MODULE` to export a dispatch factory and register `class_<YourState, ...>`.
4. **Makefile Compilation**: Compile with `-std=gnu++17` and link explicitly against `libgraph_tool_core`, `libgraph_tool_inference`, and Boost.Python with `-Wl,-rpath`.
5. **Python State Wrapper**: Subclass `DynamicsBlockStateBase`, define parameter counts/ranges in `__init__`, and override `_make_state()`.
6. **Inference & MCMC**: Use `state.entropy()` to inspect posterior description length and `state.mcmc_sweep()` to perform joint Bayesian inference over network topology and dynamical parameters.
