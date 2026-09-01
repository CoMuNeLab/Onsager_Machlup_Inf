Here is the exhaustive mathematical and algorithmic description of the inference pipeline for network reconstruction using the Minimum Description Length (MDL) principle. This guide provides the necessary formulas and step-by-step logic to implement the algorithm.

### I. The Objective Function

The core goal of the pipeline is to find the Maximum A Posteriori (MAP) estimate of the weighted adjacency matrix $W \in \mathbb{R}^{N \times N}$ given the data $X$.

$$\hat{W} = \arg\max_{W} \log P(X\vert{}W) + \log P(W)$$

.

By taking the negative logarithm, this MAP estimate is entirely equivalent to finding the weight matrix that minimizes the description length $\Sigma(X,W)$:

$$\Sigma(X,W) = -\log P(X\vert{}W) - \log P(W)$$

.

* $-\log P(X\vert{}W)$ represents the information required to encode the data given the network parameters (the likelihood).


* $-\log P(W)$ represents the information required to encode the parameters themselves (the prior, functioning as the penalty against complexity).



---

### II. Mathematical Formulation of the MDL Prior

To avoid weight shrinkage, the prior $P(W)$ is modeled using a hierarchical formulation that quantizes weights into discrete categories.

**1. Structural Variables**

* $A$: A binary adjacency matrix where $A_{ij} \in \{0, 1\}$ dictates if an edge exists.


* $E$: The total number of edges, $E = \sum_{i<j} A_{ij}$.



**2. Weight Category Variables**

* $z$: A set of $K$ discrete real-valued weight categories $z = \{z_k\}$.


* $m_k$: The count of non-zero edges assigned to category $z_k$.



**3. The Conditional Weight Prior**
The distribution of weights uniformly distributed among $E$ edges with fixed category counts $m_k$, explicitly forbidding empty categories, is defined as:

$$P(W\vert{}z,A) = \frac{\prod_k m_k!}{E!} \times \binom{E-1}{K-1}^{-1} \times \prod_{i<j} \delta_{W_{ij}, 0}^{1-A_{ij}}$$

.

**4. The Category Prior**
The weight categories $z$ themselves are penalized to prevent divergence, using a quantized Laplace distribution bounded by a precision parameter $\Delta$ (algorithmically set to a tiny value like $10^{-8}$) and a scale parameter $\lambda$ (initialized to 1):

$$P(z_k\vert{}\lambda, \Delta) \propto e^{-\lambda\vert{}z_k\vert{}}$$

.

**5. The Full Joint Prior**
By integrating all hierarchical levels (including uniform priors on $E$, $A$, and $K$), the full prior used for the description length penalty is:

$$P(W\vert{}\lambda,\Delta) = \frac{\prod_k m_k! \times e^{-\lambda \sum_k \vert{}z_k\vert{}} (e^{\lambda \Delta} - 1)^K}{E! \binom{E-1}{K-1} 2^K \max(E, 1) \binom{\binom{N}{2}}{E} (\binom{N}{2} + 1)}$$

.

---

### III. The Inference Algorithm Pipeline

Because the prior described above is discrete and non-convex, standard gradient descent cannot be used. The pipeline relies on a greedy, stochastic search algorithm that applies updates to the network iteratively, accepting moves *only* if they decrease the total description length $\Sigma(X,W)$.

**Step 1: Initialization**
Begin with an empty network where $W_{ij} = 0$ for all pairs, and an empty set of weight categories $z = \emptyset$.

**Step 2: Candidate Sub-selection**
To achieve subquadratic time complexity, avoid checking all $O(N^2)$ node pairs. Use the NNDescent algorithm to maintain and update a list of the $\kappa N$ most promising zero-valued edge candidates. Score these candidates using the pairwise ranking score $\pi_{ij}$. If $z$ is populated, calculate:

$$\pi_{ij} = \max [\ln P(W^+\vert{}X), \ln P(W^-\vert{}X)]$$

.

(Where $W^+$ and $W^-$ represent inserting the smallest positive and negative weight categories, respectively). If $z$ is empty, use the gradient magnitude instead:

$$\pi_{ij} = \left\vert{} \frac{\partial \ln P(W'\vert{}X)}{\partial W'_{ij}} \right\vert{}_{W'=W}$$

.

Let $\mathcal{E}$ be the union of these high-scoring candidates and the currently non-zero entries in $W$.

**Step 3: Edge Updates**
Iterate over every entry $(i, j)$ in $\mathcal{E}$ and propose one of the following moves uniformly at random:

* Assign $W'_{ij} \in z$: Choose an existing category via random bisection search to maximize the objective.


* Create a new category $W'_{ij} \notin z$: Perform a random bisection search in the continuous allowed weight space, and append this new value to $z$.


* Deletion: If $W_{ij} \neq 0$, propose setting $W'_{ij} \leftarrow 0$.



**Step 4: Edge Moves**
Iterate over nodes $i$ with a non-zero degree. Pick a random incident edge $(i, j)$ and a random zero-valued candidate $(i, u) \in \mathcal{E}$. Swap their values: $W'_{iu} \leftarrow W_{ij}$ and $W'_{ij} \leftarrow 0$.

**Step 5: Edge Swaps**
Select two distinct non-zero entries $(i, j)$ and $(u, v)$ uniformly at random. Swap their endpoints so that the new edges are $(i, v)$ and $(u, j)$, transferring the respective weights.

**Step 6: Weight Category Value Updates**
Iterate over every existing category $z_k \in z$. Perform a continuous bisection search over allowed values to find a new optimum $z'_k$ and update all edges currently assigned to that category.

**Step 7: Merge-Split Clustering for Categories**
Optimize the distribution of categories to eliminate redundant bins:

* **Merge**: Propose combining two categories $z_k$ and $z_l$ into a single new category $z_m$, finding the optimal value for $z_m$ via bisection search.


* **Split**: Propose dividing a category $z_m$ into two new bins $z_k$ and $z_l$ (initialized randomly near $z_m$). Redistribute the edges between them and optimize their values via bisection search until convergence.


* **Merge-Split**: Execute a merge immediately followed by a split to reshuffle edges into two new optimum categories without changing the total $K$.



**Step 8: Convergence**
Repeat Steps 2 through 7 continually in arbitrary order. Terminate the pipeline when a full sweep of these operations results in no further improvement (decrease) to the description length $\Sigma(X,W)$.