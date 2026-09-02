# Entropy Investigation Results & Root Cause Analysis

I wrote and executed the diagnostic script to test our hypotheses regarding the True Network's entropy. The results fully uncovered the mystery behind the entropy paradox. 

Here are the key findings from the experiments:

## 1. The Missing Edge Weights (The Root Cause)
When evaluating `S_true`, we passed the true graph structure `g=g_true` to `GeneExprBlockState`. However, we discovered a crucial quirk in the `graph-tool` extension: **passing the graph structure does not automatically pass the edge weights.** 

Because we did not explicitly pass `x=g_true.ep["x"]` in the constructor, `graph-tool` was initializing the "True Network" with randomized or zeroed edge weights!
- The true structure was evaluated with completely incorrect interaction strengths.
- The predicted drift for every gene was completely wrong.
- The likelihood squared error exploded, leading to a massive penalty (e.g., `S_true = 127658`).

When properly initialized with `x=g_true.ep["x"]`, the True Network correctly predicts the dynamics.

## 2. The Discretization Penalty (`dt_meas = 0.5`)
Even when properly evaluated, there is a second major force at play: **discretization error**. 
In our experiments, we saw that the MCMC algorithm was finding networks with incredibly negative (i.e., highly optimal) entropies compared to the baseline. Why?
- The simulation data has continuous curvature over a 0.5s interval. 
- The likelihood function assumes a linear Euler step over that 0.5s.
- To minimize this approximation error, the MCMC optimizer **artificially alters the parameters** (which we saw in the printed `Inferred Theta` arrays) and **drops/modifies edges** to create an "effective" network that perfectly compensates for the Euler integration error. 
- This is why the inferred network can score better than the true network on coarsely sampled data: it is cheating the approximation!

## 3. Fixed Theta Experiment (Network Only)
As requested, I ran an experiment attempting to freeze the kinetic parameters ($\theta$) to the true values and only update the network structure/weights.
- The MCMC still preferred a drastically sparser graph (**3 edges inferred** vs 18 true edges).
- Because the parameters were anchored to the "true continuous" values, they couldn't be altered to compensate for the 0.5s discretization error. 
- The resulting prediction errors were so punishing that the MCMC aggressively pruned the true edges to save on structural complexity costs. It determined that having no edges was statistically safer than having edges that incur discretization penalties!

## Conclusion & How to Fix Your Notebook
To fix the baseline true entropy in your `custom_grn_inference.ipynb` notebook, you simply need to explicitly pass the edge weights when instantiating the true state.

Change this:
```python
state_true = GeneExprBlockState(s, dt=dt_meas, g=g_true, theta=theta_true)
```
To this:
```python
state_true = GeneExprBlockState(s, dt=dt_meas, g=g_true, theta=theta_true, x=g_true.ep["x"])
```

Once this is fixed, `S_true` will properly reflect the true parameters and weights. Note that on coarsely sampled data (`dt_meas=0.5`), the MCMC will likely still find a "better" effective network to cheat the integration error, but the true network will no longer be artificially penalized by randomized weights!
