import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results_information_content.csv")

# Filter for stationary regime and reasonable dt_meas to avoid low resolution artifacts
stat_df = df[(df['Regime'] == 'Stationary') & (df['dt_meas'] <= 0.5)]

# Group by parameters to find the most informative settings
grouped = stat_df.groupby(['Topology', 'd', 'V', 'Noise', 'dt_meas'])[['TDMI', 'eSNR', 'TIC', 'Var']].mean().reset_index()

# Sort by TIC descending
top_settings = grouped.sort_values(by='TIC', ascending=False)
print("Top 10 parameter settings for Stationary Information Content:")
print(top_settings.head(10).to_string())

# Save the top setting
best = top_settings.iloc[0]
print("\nBest setting:", dict(best))
