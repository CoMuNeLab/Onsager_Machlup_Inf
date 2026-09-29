md_append = """
## Result Visualizations

The following plots visualize the mean and standard deviation across the 3 independent trials for each scenario.

![F1-Score vs Variance](/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/images/f1_vs_variance.png)
<br/>

![False Positives vs Variance](/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/images/fp_vs_variance.png)
<br/>

![Entropy Gap vs Variance](/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/images/entropy_gap_vs_variance.png)
"""

with open("/home/francesco/.gemini/antigravity/brain/f567a5f8-b885-4710-bb5e-c6bf2233a929/high_variance_walkthrough.md", "a") as f:
    f.write(md_append)

