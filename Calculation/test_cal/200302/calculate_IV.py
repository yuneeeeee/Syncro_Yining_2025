import pandas as pd
import numpy as np
from scipy import stats
from scipy import optimize
from sklearn import linear_model
import os
import glob
import re

# v_steps = []

def get_sweep_range(df, column_pattern="Sweep *"):
    sweep_columns = [col for col in df.columns if re.match(column_pattern, col[0])]
    sweep_indices = sorted(set(int(re.search(r"\d+", col[0]).group()) for col in sweep_columns if re.search(r"\d+", col[0])))
    return sweep_indices

# def initialize_v_steps(df):
#     global v_steps
#     sweep_indices = get_sweep_range(df)
#     if not sweep_indices:
#         print("Error: No sweep indices found.")
#         return
    
#     first_sweep = df.at['Sweep Voltage/Current', f'Sweep {sweep_indices[0]:03d}'] * 1000
#     last_sweep = df.at['Sweep Voltage/Current', f'Sweep {sweep_indices[-1]:03d}'] * 1000
    
#     num_sweeps = len(sweep_indices)
#     step_size = (last_sweep - first_sweep) / (num_sweeps - 1) if num_sweeps > 1 else 1
    
#     v_steps = [first_sweep + i * step_size for i in range(num_sweeps)]

# Function to filter data by `Valid QC = 1`
def filter_valid_qc(df):
    # Identify Valid QC column in the second row of the headers
    valid_qc_col = next((col for col in df.columns if col[1].strip() == 'Valid QC'), None)
    if valid_qc_col:
        df_filtered = df[~df[valid_qc_col].astype(str).str.strip().str.upper().eq('F')]
        return df_filtered
    else:
        print("Warning: 'Valid QC' column not found. Proceeding with all wells.")
        return df

# Function to process Max Current Density
def calculate_max_cd(df):
    sweep_indices = get_sweep_range(df)
    tp1_curden_columns = [(f'Sweep {i:03d}', 'TP1CurDen') for i in sweep_indices]
    tp2_curden_columns = [(f'Sweep {i:03d}', 'TP2CurDen') for i in sweep_indices]
    Parameter_column = ('Sweep Results', 'Parameter')

    all_columns = [Parameter_column] + tp1_curden_columns + tp2_curden_columns
    df_extracted = df.loc[:, all_columns]
    df_extracted.columns = ['Parameter'] + [f'TP1CurDen_{i}' for i in sweep_indices] + [f'TP2CurDen_{i}' for i in sweep_indices]

    def get_largest_absolute(df, column_prefix, sweep_indices):
        columns = [f'{column_prefix}_{i}' for i in sweep_indices]
        return df[columns].apply(lambda row: row.loc[row.abs().idxmax()] if not row.isna().all() else np.nan, axis=1)

    df_extracted['TP1_Max_CurDen'] = get_largest_absolute(df_extracted, 'TP1CurDen', sweep_indices)
    df_extracted['TP2_Max_CurDen'] = get_largest_absolute(df_extracted, 'TP2CurDen', sweep_indices)
    return df_extracted[['Parameter', 'TP1_Max_CurDen', 'TP2_Max_CurDen']]

# Function to process Tau Inactivation
def calculate_tau_inact(df):
    v_steps = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
    sweep_indices = get_sweep_range(df)

    def get_tau_inact_at_0mV(df, input_column, v_steps):
        tau_columns = [(f'Sweep {i:03d}', input_column) for i in sweep_indices]
        df_extracted = df.loc[:, [('Sweep Results', 'Parameter')] + tau_columns]
        df_extracted.columns = ['Parameter'] + [f'Tau_{i}' for i in sweep_indices]
        df_extracted_num = df_extracted.apply(pd.to_numeric, errors='coerce')

        tau_inact_at_0mV_list = []
        Parameter_list = []
        for cell_index in df_extracted.index:
            tau_values = df_extracted_num.loc[cell_index, f'Tau_{sweep_indices[0]}':f'Tau_{sweep_indices[-1]}'].values
            tau_subset = tau_values[10:15]
            if np.any(np.isnan(tau_subset)):
                Parameter_list.append(df_extracted.loc[cell_index, 'Parameter'])
                tau_inact_at_0mV_list.append(np.nan)
                continue
            log_reciprocal = np.log(1 / tau_subset)
            slope, intercept, *_ = stats.linregress(v_steps[10:15], log_reciprocal)
            tau_at_0mV = 1000 / np.exp(intercept)
            tau_inact_at_0mV_list.append(tau_at_0mV)
            Parameter_list.append(df_extracted.loc[cell_index, 'Parameter'])

        return Parameter_list, tau_inact_at_0mV_list

    tau1_param, tau1_values = get_tau_inact_at_0mV(df, 'TP1Tau1In', v_steps)
    tau2_param, tau2_values = get_tau_inact_at_0mV(df, 'TP1Tau2In', v_steps)
    return pd.DataFrame({
        'Parameter': tau1_param,
        'TP1Tau1In_at_0mV': tau1_values,
        'TP1Tau2In_at_0mV': tau2_values
    })

def calculate_v_half(df):
    def func_boltzmann(x, a, b, c, d):
        return (a + ((b - a) / (1 + np.exp((c - x) / d))))

    v_steps = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
    sweep_indices = get_sweep_range(df)

    def fit_boltzmann(g_values, v_steps):
        try:
            popt, _ = optimize.curve_fit(func_boltzmann, v_steps, g_values, maxfev=5000)
            return popt[2], popt[3]  # v_half and slope
        except:
            return np.nan, np.nan

    g_columns = [(f'Sweep {i:03d}', 'TP1PeakCur') for i in sweep_indices]
    df_extracted = df.loc[:, [('Sweep Results', 'Parameter')] + g_columns]
    df_extracted.columns = ['Parameter'] + [f'G_{i}' for i in sweep_indices]
    v_half_values = []
    slope_values = []

    for idx in df_extracted.index:
        # Convert g_values to numeric and handle non-numeric values
        g_values = pd.to_numeric(df_extracted.loc[idx, f'G_{sweep_indices[0]}':f'G_{sweep_indices[-1]}'], errors='coerce').values
        if np.any(np.isnan(g_values)):  # Check for NaNs after conversion
            v_half_values.append(np.nan)
            slope_values.append(np.nan)
        else:
            v_half, slope = fit_boltzmann(g_values, v_steps)
            v_half_values.append(v_half)
            slope_values.append(slope)

    df_extracted['V_half'] = v_half_values
    df_extracted['Slope'] = slope_values
    return df_extracted[['Parameter', 'V_half', 'Slope']]


# Function to extract compound name
def get_compound_name(df):
    df_extracted = df.loc[:, [('Sweep Results', 'Parameter'), ('Sweep 001', 'Compound Name')]]
    df_extracted.columns = ['Parameter', 'Compound Name']
    return df_extracted

# Consolidated processing function
def process_file(input_path):
    df = pd.read_csv(input_path, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples([(col1.strip(), col2.strip()) for col1, col2 in df.columns])

    # Filter wells with `Valid QC = 1`
    df_filtered = filter_valid_qc(df)

    # Calculate individual results
    max_cd_results = calculate_max_cd(df_filtered)
    tau_inact_results = calculate_tau_inact(df_filtered)
    v_half_results = calculate_v_half(df_filtered)
    compound_names = get_compound_name(df_filtered)

    # Merge results
    merged_results = pd.merge(max_cd_results, tau_inact_results, on='Parameter', how='outer')
    merged_results = pd.merge(merged_results, v_half_results, on='Parameter', how='outer')
    merged_results = pd.merge(merged_results, compound_names, on='Parameter', how='left')

    # Output file path
    output_path = input_path.replace('.csv', '_calculated.csv')
    merged_results.to_csv(output_path, index=False)
    print(f"Processed {input_path} and saved results to {output_path}")

# Get all input files and process them
input_files = glob.glob("*IV*.csv")
for input_file in input_files:
    process_file(input_file)

print("All files processed.")
