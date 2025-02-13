import pandas as pd
import numpy as np
from scipy import stats
import glob


# Function to calculate Tau Closing at 0mV
def get_tau_closing_at_0mV_fixed(df_input, tau_columns):
    v_steps = [-100, -90, -80, -70, -60, -50, -40]
    tau_closing_at_0mV_list = []

    for cell_index in df_input.index:
        # Extract Tau values for all sweeps
        tau_closing_values = df_input.loc[cell_index, tau_columns].values.astype(float)
        tau_closing_n100_to_n70 = tau_closing_values[:4]

        if np.any(np.isnan(tau_closing_n100_to_n70)):  # Skip if NaN exists
            tau_closing_at_0mV_list.append(np.nan)
            continue

        # Perform linear regression on log-transformed reciprocal values
        tau_closing_n100_to_n70_log_reciprocal = np.log(1 / tau_closing_n100_to_n70)
        slope, intercept, *_ = stats.linregress(v_steps[:4], tau_closing_n100_to_n70_log_reciprocal)
        tau_closing_at_0mV = 1 / np.exp(intercept)
        tau_closing_at_0mV_list.append(tau_closing_at_0mV * 1000)  # Convert to ms

    return tau_closing_at_0mV_list


# Function to process a single file
def process_tail_file(file_path):
    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path, header=[0, 1])

    # Strip whitespace from column names
    df.columns = pd.MultiIndex.from_tuples([(col1.strip(), col2.strip()) for col1, col2 in df.columns])

    # Filter rows where Valid QC == 1
    if ('Unnamed: 1_level_0', 'Valid QC') in df.columns:
        df = df[df[('Unnamed: 1_level_0', 'Valid QC')] == 1]
    else:
        print(f"'Valid QC' column not found in {file_path}. Proceeding with all wells.")

    # Dynamically find columns for Compound Name, TailTau1, and TailTau2
    compound_name_columns = [col for col in df.columns if 'Compound Name' in col[1]]
    tau1_columns = [col for col in df.columns if 'TailTau1' in col[1]]
    tau2_columns = [col for col in df.columns if 'TailTau2' in col[1]]
    
    # Dynamically find columns for Compound Name, TailTau1, and TailTau2
    print("Inspecting columns in the DataFrame:")
    print(df.columns.tolist())  # Print all column names for inspection

    compound_name_columns = [col for col in df.columns if col[1] == 'Compound Name']
    tau1_columns = [col for col in df.columns if col[1] == 'TailTau1']
    tau2_columns = [col for col in df.columns if col[1] == 'TailTau2']
    # Debug: Print matched columns
    # print("\nMatched columns:")
    # print(f"Compound Name columns: {compound_name_columns}")
    # print(f"Tau1 columns: {tau1_columns}")
    # print(f"Tau2 columns: {tau2_columns}")


    # Ensure we have exactly 7 Tau columns for each type
    if len(tau1_columns) != 7 or len(tau2_columns) != 7:
        print(f"Missing or inconsistent Tau columns in {file_path}. Skipping file.")
        return

    # Extract necessary columns
    columns_to_extract = [('Sweep Results', 'Parameter')] + compound_name_columns + tau1_columns + tau2_columns
    df_extracted = df.loc[:, columns_to_extract]

    # Calculate Tau Closing at 0mV for TailTau1 and TailTau2
    tau1_closing_list = get_tau_closing_at_0mV_fixed(df_extracted, tau1_columns)
    tau2_closing_list = get_tau_closing_at_0mV_fixed(df_extracted, tau2_columns)

    # Create output DataFrame with results
    output_df = pd.DataFrame({
        'Well_ID': df_extracted[('Sweep Results', 'Parameter')],
        'Compound_Name': df_extracted[compound_name_columns[0]],
        'Tau1_closing_at_0mV': tau1_closing_list,
        'Tau2_closing_at_0mV': tau2_closing_list
    })

    # Save results to a new CSV file
    output_csv_path = file_path.replace('.csv', '_Tau_closing_at_0mV.csv')
    output_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


# Process all *Tail*.csv files in the current folder
tail_files = glob.glob("*Tail*.csv")
for tail_file in tail_files:
    process_tail_file(tail_file)

print("Processing complete.")
