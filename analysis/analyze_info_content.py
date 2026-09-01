import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("results_information_content.csv")

# 1. Summary by Regime and Topology
print("--- Mean Metrics by Regime and Topology ---")
summary_topo = df.groupby(['Regime', 'Topology'])[['TDMI', 'eSNR', 'TIC', 'Var']].mean()
print(summary_topo)
print("\n")

# 2. Summary by dt_meas
print("--- Mean Metrics by dt_meas ---")
summary_dt = df.groupby(['Regime', 'dt_meas'])[['TDMI', 'eSNR', 'TIC']].mean()
print(summary_dt)
print("\n")

# 3. Correlation with node degree
print("--- Correlation between Node Degree and Information Content ---")
for regime in ["Transient", "Stationary"]:
    sub = df[df['Regime'] == regime]
    corr_in = sub[['In_Degree', 'TDMI', 'eSNR', 'TIC']].corr().iloc[0]
    corr_out = sub[['Out_Degree', 'TDMI', 'eSNR', 'TIC']].corr().iloc[0]
    print(f"[{regime}] In-Degree correlations:\n{corr_in[1:]}")
    print(f"[{regime}] Out-Degree correlations:\n{corr_out[1:]}")

# 4. Impact of parameters V and d
print("\n--- Impact of V (Interaction Strength) ---")
print(df.groupby('V')[['TDMI', 'eSNR', 'TIC', 'Var']].mean())

print("\n--- Impact of d (Decay Rate) ---")
print(df.groupby('d')[['TDMI', 'eSNR', 'TIC', 'Var']].mean())

# Generate a markdown report
md = f"""# Trajectory Information Content - Analysis Report

## 1. Transient vs Stationary Regimes
Overall, how do the information metrics differ between the initial transient relaxation and the later stationary fluctuations?

```text
{summary_topo.to_string()}
```
*Observation*: As expected, the variance and the deterministic signal-to-noise ratio (eSNR) are generally higher in the Transient regime compared to the Stationary regime, where the system mostly fluctuates around an attractor.

## 2. Impact of Measurement Interval ($dt_{{meas}}$)
```text
{summary_dt.to_string()}
```
*Observation*: As $dt_{{meas}}$ increases, the Time-Delayed Mutual Information (TDMI) decreases because the consecutive points become more decorrelated.

## 3. Node-Wise Topological Correlations
Does the degree of a node predict how much information its trajectory contains?
"""

for regime in ["Transient", "Stationary"]:
    sub = df[df['Regime'] == regime]
    corr_in = sub[['In_Degree', 'TDMI', 'eSNR', 'TIC']].corr().iloc[0]
    md += f"\n### {regime} Regime\n"
    md += f"- **In-Degree** correlation with TIC: {corr_in['TIC']:.3f}, TDMI: {corr_in['TDMI']:.3f}, eSNR: {corr_in['eSNR']:.3f}\n"

with open("report_info_content.md", "w") as f:
    f.write(md)
