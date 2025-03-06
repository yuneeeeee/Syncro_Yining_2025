import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(manual_file_path, auto_file_path):
    # Load the uploaded files
    manual_df = pd.read_csv(manual_file_path)
    auto_df = pd.read_csv(auto_file_path)

    # Find the starting column: 'TP1_Max_CurDen'
    # start_index = manual_df.columns.get_loc('TP1_Max_CurDen')
    start_index = manual_df.columns.get_loc('Rec_tau')

    # Select relevant columns from the starting point onwards and exclude 'TP1Tau2In_at_0mV'
    manual_selected = manual_df.iloc[:, start_index:]
    auto_selected = auto_df.iloc[:, start_index:]

    manual_selected = manual_selected.drop(columns=['Rec_A0','Rec_Aend'], errors='ignore')
    auto_selected = auto_selected.drop(columns=['Rec_A0','Rec_Aend'], errors='ignore')

    # Calculate mean and standard deviation for each column
    manual_stats = manual_selected.agg(['mean', 'std']).transpose()
    auto_stats = auto_selected.agg(['mean', 'std']).transpose()

    # Prepare data for combined bar plot
    labels = manual_stats.index
    manual_means = manual_stats['mean'].values
    manual_stds = manual_stats['std'].values
    auto_means = auto_stats['mean'].values
    auto_stds = auto_stats['std'].values

    x = range(len(labels))

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Plot
    plt.figure(figsize=(18, 8))
    bar_width = 0.35
    colors = sns.color_palette("muted", 2)

    bars1 = plt.bar([i - bar_width/2 for i in x], manual_means, yerr=manual_stds, capsize=5,
                    width=bar_width, label='Manual', color=colors[0], alpha=0.8)
    bars2 = plt.bar([i + bar_width/2 for i in x], auto_means, yerr=auto_stds, capsize=5,
                    width=bar_width, label='Auto', color=colors[1], alpha=0.8)

    # Label each bar with the mean value
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(x, labels, rotation=90, fontsize=10)
    plt.ylabel('Value', fontsize=12)
    plt.title('-100 Tail test data comparison', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Example usage
plot_comparison('SPM 10.04.2418T27331_Cav3_Recovery_-100mV_manual_Recovery_summary.csv',
                'SPM 10.04.2418T27331_Cav3_Recovery_-100mV_auto_Recovery_summary.csv')
