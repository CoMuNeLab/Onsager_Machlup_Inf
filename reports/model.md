# The model (ODE form)

We model the time evolution of the expression levels of $N$ genes as

$
    \frac{dx_i(t)}{dt}=-\gamma_i x_i(t)+V_i\cdot\;\sigma\Big( \sum_j \, W_{i,j}\,x_j\Big) + \sqrt{2D_i}\, \xi_i(t)
$

where $x_i(t)$ is the expression level of the $i$-th gene, $\gamma_i$ is the decay rate, $V_i$ is the maximum expression level, $W_{i,j}$ is the interaction strength, and $\xi_i(t)$ is Gaussian white noise. The nonlinear activation function is defined as $\sigma(x) = \frac{1}{1+e^{-x}}$.

The learnable parameters are:
- in the node-level
    - the decay rates $\gamma_i$
    - the maximum expression levels $V_i$
    - the diffusion coefficients $D_i$
- in the edge-level
    - the interaction strengths $W_{i,j}$


# Likelihood

We work with discrete time measurements $t_0, t_1, \dots, t_M$, with $\Delta t = t_{k+1}-t_k$. We assume that the initial state $x(t_0)$ is known. The conditional probability density of the trajectory $x(t_0:t_M)$ under the SDE is given by the Onsager-Machlup functional 


$P(\{\mathbf{x}_k\}_{k=1}^M|\Theta)=\Big(\frac{1}{\sqrt{2\pi D\Delta t}}\Big)^M\exp\Big[-\frac{\Delta t}{2D}\sum_{k=1}^M\Big(\frac{\mathbf{x}_k-\mathbf{x}_{k-1}}{\Delta t}-\mathbf{f}(\mathbf{x}_{k-1}|\Theta)\Big)^2\Big]+O(\Delta t^2)$

where $\mathbf{f}(\mathbf{x})$ is the drift term of the SDE.

The time interval $\Delta t$ is assumed to be small enough such that the SDE can be approximated by the Euler-Maruyama scheme, and it is given in the inference model definition phase. 