import pandas as pd
import numpy as np

def analyze_grid(csv_path, name):
    try:
        df = pd.read_csv(csv_path)
        print(f"=== Analysis for {name} ===")
        print(f"Total experiments: {len(df)}")
        
        # Look at mean AUPRC Ratio and AUPRC by State
        if 'State' in df.columns and 'AUPRC_Ratio' in df.columns:
            print("\nMean AUPRC Ratio by State:")
            print(df.groupby('State')['AUPRC_Ratio'].mean())
            
        # Look at effect of network size (N) on performance
        if 'N' in df.columns:
            print("\nMean AUPRC Ratio by N:")
            print(df.groupby('N')['AUPRC_Ratio'].mean())
            print("\nMean Spearman by N:")
            print(df.groupby('N')['Spearman'].mean())
            
        print("\nOverall Max AUPRC_Ratio:", df['AUPRC_Ratio'].max())
        print("Overall Mean AUPRC_Ratio:", df['AUPRC_Ratio'].mean())
        print("Overall Mean Spearman:", df['Spearman'].mean())
        print("-" * 50)
        return df
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

df_sf = analyze_grid('results/grid_results/results/grid_results/performance_metrics.csv', 'Scale-Free')
df_er = analyze_grid('results/grid_results_er/results/grid_results_er/performance_metrics.csv', 'Erdos-Renyi')
