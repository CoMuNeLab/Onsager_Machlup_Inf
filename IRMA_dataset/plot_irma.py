import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # Convert time to numeric, mapping Glc to 0 and Gal 0 to 10
    time_map = {'Glc': 0, 'Gal 0': 10}
    
    def parse_time(val):
        if str(val) in time_map:
            return time_map[str(val)]
        try:
            return float(val)
        except:
            return None

    df['Time_numeric'] = df['Time_min'].apply(parse_time)
    df = df.dropna(subset=['Time_numeric'])
    
    genes = ['CBF1', 'GAL4', 'SWI5', 'GAL80', 'ASH1']
    return df['Time_numeric'], df[genes]

def plot_trajectories():
    t_on, expr_on = load_and_preprocess('irma_switch_on.csv')
    t_off, expr_off = load_and_preprocess('irma_switch_off.csv')
    
    genes = expr_on.columns
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Switch-On
    for gene in genes:
        axes[0].plot(t_on, expr_on[gene], marker='o', label=gene)
    axes[0].set_title('IRMA Switch-On Trajectories')
    axes[0].set_ylabel('Expression Level')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Switch-Off
    for gene in genes:
        axes[1].plot(t_off, expr_off[gene], marker='o', label=gene)
    axes[1].set_title('IRMA Switch-Off Trajectories')
    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Expression Level')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'irma_trajectories.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_trajectories()
