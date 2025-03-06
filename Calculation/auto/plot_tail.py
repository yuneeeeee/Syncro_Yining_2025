import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_concentration_vs_tau(file_path):
    # Load the summary file
    df = pd.read_csv(file_path)
    
    # Filter to ensure valid values
    df = df[['Concentration', 'TP1Tau1In_at_0mV']].dropna()
    
    # Treat Concentration as a string to ensure equal spacing
    df['Concentration'] = df['Concentration'].astype(str)
    grouped = df.groupby('Concentration')['TP1Tau1In_at_0mV'].agg(['mean', 'std']).reset_index()
    
    # Plot with equal spacing between x-axis groups
    plt.figure(figsize=(8, 6))
    x_positions = np.arange(len(grouped))  # Evenly spaced x positions
    plt.errorbar(x_positions, grouped['mean'], yerr=grouped['std'], fmt='o-', capsize=5, label='TP1Tau1In_at_0mV')
    plt.xticks(x_positions, grouped['Concentration'])
    plt.xlabel('Concentration')
    plt.ylabel('TP1Tau1In_at_0mV')
    plt.title('TP1Tau1In_at_0mV vs. Concentration')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
plot_concentration_vs_tau('SPM 10.04.2418T27331_Cav3 IV_-75mV_auto_summary.csv')
