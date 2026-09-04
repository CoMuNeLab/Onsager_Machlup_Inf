# Exact Likelihood Evaluation for Discretely Sampled Gene Regulatory Networks

## 1. The Need for Trajectory Likelihoods

In the context of Bayesian network inference—such as Markov Chain Monte Carlo (MCMC) optimization—reconstructing a Gene Regulatory Network (GRN) from time-series data requires a rigorous method to score proposed network configurations. Specifically, given a proposed adjacency matrix and kinetic parameter set $\Theta$, the algorithm must evaluate the likelihood of observing the discretely sampled gene expression trajectories $\mathbf{X}$.

Let $x_i(t_k)$ denote the expression level of gene $i$ observed at discrete time points $t_0, t_1, \dots, t_M$, separated by a measurement interval $\Delta t$. We model the underlying continuous biological dynamics using a system of stochastic differential equations (SDEs):

$$ dx_i(t) = \Big(-\gamma_i x_i(t) + V_i \sigma(m_i(t))\Big) dt + \sqrt{D_i x_i(t)} dW_i(t) $$

where $\gamma_i$ is the intrinsic degradation rate, $V_i$ is the maximum transcription rate, and $m_i(t) = \sum_j W_{ij} x_j(t)$ represents the aggregate regulatory input from the network mapped through a non-linear activation function $\sigma(\cdot)$. The noise term is modeled as multiplicative Cox-Ingersoll-Ross (CIR) noise, scaled by a diffusion coefficient $D_i$ and driven by a standard Wiener process $W_i(t)$.

To compute the total likelihood of a trajectory, the process is treated as a Markov chain, requiring the evaluation of the transition probability $P(x_{i, t+\Delta t} \mid \mathbf{X}_t, \Theta)$ for every step.

## 2. Limitations of Standard Discretization

Traditionally, this transition probability is approximated using the Euler-Maruyama numerical scheme. This approach assumes that the entire drift derivative remains constant over the measurement interval $\Delta t$, yielding a Gaussian transition density:

$$ P(x_{t+\Delta t} \mid \mathbf{X}_t) \sim \mathcal{N}\left( x_t + \Delta t \big(-\gamma x_t + V\sigma(m_t)\big), \; D x_t \Delta t \right) $$

While computationally trivial, this approximation introduces catastrophic structural bias when applied to sparse biological datasets. Real-world transcriptomic time-series often exhibit large measurement intervals (e.g., hours between sampling). Under large $\Delta t$, the Euler-Maruyama linear tangent completely ignores the exponential relaxation inherent to the $-\gamma x_t$ degradation term. This results in severe mathematical overshoot, where the predicted mean may drop below zero or artificially inflate.

Consequently, during Bayesian optimization, the MCMC algorithm heavily penalizes the true network structure due to these numerical integration errors. The optimizer is forced to artificially sparsify the network or distort kinetic parameters simply to compensate for the discretization penalty—a phenomenon that breaks the generative fidelity of the model.

## 3. The Zero-Order Hold Assumption

To eliminate discretization error without incurring the computationally intractable cost of inferring unobserved latent variables between measurement points, we introduce an exact analytical integration approach predicated on a **Zero-Order Hold** assumption.

We assume that the aggregate regulatory input from the rest of the network, $m_i(t)$, remains piecewise constant over the short interval $[t, t+\Delta t]$. Under this assumption, the non-linear coupling term $V_i \sigma(m_i(t))$ simplifies to a constant scalar $b$.

The governing SDE for a single gene isolated during this time step reduces to a strictly linear stochastic differential equation:

$$ dx_t = (-\gamma x_t + b) dt + \sqrt{D x_t} dW_t $$

By isolating the gene's state variable $x_t$, the SDE is mathematically equivalent to a continuous CIR process, which permits an exact analytical solution for its transition moments regardless of the magnitude of $\Delta t$.

## 4. Analytical Derivation of Exact Transition Moments

Because the isolated SDE is linear with respect to the state variable $x_t$, it can be solved exactly using an integrating factor $e^{\gamma t}$ and Itô's Lemma. Applying Itô's Lemma to the product $x_t e^{\gamma t}$ yields:

$$ d(x_t e^{\gamma t}) = b e^{\gamma t} dt + e^{\gamma t} \sqrt{D x_t} dW_t $$

Integrating this expression from $0$ to $\Delta t$ provides the exact analytical path of the gene expression:

$$ x_{t+\Delta t} = \underbrace{x_t e^{-\gamma \Delta t} + \frac{b}{\gamma} (1 - e^{-\gamma \Delta t})}_{\text{Deterministic Mean } (\mu)} + \underbrace{\int_0^{\Delta t} e^{-\gamma (\Delta t - s)} \sqrt{D x_s} dW_s}_{\text{Stochastic Noise}} $$

### 4.1 Exact Expected Mean
The deterministic component of the solution perfectly captures the continuous exponential relaxation of the gene. The expected mean exponentially decays toward a steady-state asymptote defined by the regulatory input, $\mu_\infty = \frac{b}{\gamma} = \frac{V \sigma(m_t)}{\gamma}$. 
Therefore, the exact transition mean is:

$$ \mathbb{E}[x_{t+\Delta t} \mid x_t] = \mu_\infty + (x_t - \mu_\infty)e^{-\gamma \Delta t} $$

### 4.2 Exact Integrated Variance via Itô's Isometry
To calculate the exact variance accumulated over the interval $\Delta t$, we must find the variance of the stochastic Itô integral. By applying **Itô's Isometry**, the variance of the stochastic integral transforms into an ordinary Riemann integral of the expected state trajectory:

$$ \text{Var}\left( \int_0^{\Delta t} e^{-\gamma (\Delta t - s)} \sqrt{D x_s} dW_s \right) = \mathbb{E}\left[ \int_0^{\Delta t} e^{-2\gamma(\Delta t - s)} D x_s \, ds \right] $$

Because the expectation operator is linear, we substitute the exact expected mean $\mathbb{E}[x_s \mid x_t]$ derived in Section 4.1 into the integral. Evaluating this integral yields a closed-form analytical variance:

$$ \text{Var}(x_{t+\Delta t} \mid x_t) = \frac{D}{\gamma}(x_t - \mu_\infty)\big(e^{-\gamma \Delta t} - e^{-2\gamma \Delta t}\big) + \frac{D \mu_\infty}{2\gamma}\big(1 - e^{-2\gamma \Delta t}\big) $$

This exact variance formula exhibits perfect asymptotic behavior: as $\Delta t \to 0$, it elegantly collapses to the instantaneous Euler variance $D x_t \Delta t$, and as $\Delta t \to \infty$, it strictly bounds to the steady-state variance $\frac{D \mu_\infty}{2\gamma}$, preventing the likelihood from diverging.

## 5. Summary

By replacing the Euler-Maruyama approximations with these exact analytical moments, the Gaussian transition likelihood $P(x_{i, t+\Delta t} \mid \mathbf{X}_t) \sim \mathcal{N}(\mu, \text{Var})$ evaluates the trajectory with exact physical fidelity. The structural bias introduced by numerical discretization is mathematically eliminated, allowing the Bayesian inference algorithm to identify the true biological network topology even across highly coarse, real-world sampling intervals.
