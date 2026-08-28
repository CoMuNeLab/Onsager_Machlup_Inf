import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory for saving plots
out_dir = "/home/francesco/.gemini/antigravity/brain/bfdc093e-0921-4ff2-a18c-45b223faec92"

# Load data
df = pd.read_csv("results_information_content.csv")

# Ensure seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Plot 1: Regime Comparison (Boxplots of TIC and TDMI)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df, x="Regime", y="TDMI", hue="Topology", ax=axes[0])
axes[0].set_title("Time-Delayed Mutual Info (TDMI) by Regime")
axes[0].set_ylabel("TDMI (nats)")

sns.boxplot(data=df, x="Regime", y="TIC", hue="Topology", ax=axes[1])
axes[1].set_title("Trajectory Information Capacity (TIC) by Regime")
axes[1].set_ylabel("TIC")
axes[1].set_yscale("log") # log scale for TIC

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_regime_comparison.png"), dpi=150)
plt.close()

# Plot 2: Impact of Measurement Interval (dt_meas)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="dt_meas", y="TDMI", hue="Regime", marker="o", errorbar=('ci', 95))
plt.title("Impact of Measurement Interval on Information Content")
plt.xlabel(r"Measurement Interval ($dt_{meas}$)")
plt.ylabel("TDMI")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_dt_meas_impact.png"), dpi=150)
plt.close()

# Plot 3: Node Degree vs Information Content
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="In_Degree", y="TIC", hue="Regime", alpha=0.5, edgecolor=None)
plt.title("Information Content (TIC) vs Node In-Degree")
plt.xlabel("Node In-Degree")
plt.ylabel("TIC (log scale)")
plt.yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_degree_correlation.png"), dpi=150)
plt.close()

# Plot 4: Impact of Physical Parameters (V and d)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=df, x="V", y="TIC", hue="Regime", ax=axes[0])
axes[0].set_title("Impact of Interaction Strength (V)")
axes[0].set_xlabel("V (Strength)")
axes[0].set_ylabel("TIC")

sns.barplot(data=df, x="d", y="TIC", hue="Regime", ax=axes[1])
axes[1].set_title("Impact of Decay Rate (d)")
axes[1].set_xlabel("d (Decay Rate)")
axes[1].set_ylabel("TIC")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot_parameters_impact.png"), dpi=150)
plt.close()

print("Plots successfully generated and saved to the artifact directory.")
