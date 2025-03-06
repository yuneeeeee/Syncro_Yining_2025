import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

v_step = [-100, -90, -80, -70, -60, -50, -40]

def extract_meta_columns(filepath):
    df = pd.read_csv(filepath, header=[0, 1])
    second_header = df.columns.get_level_values(1).str.strip()
    meta_columns_to_extract = ['Experiment  full  Name', 'Experiment Date', 'Well ID', 'Valid QC', 'Cell Type', 'Cell Concentration']
    meta_cols = [col for col, name in zip(df.columns, second_header) if name in meta_columns_to_extract]
    df_meta = df[meta_cols]
    df_meta.columns = meta_columns_to_extract
    if 'Valid QC' in df_meta.columns:
        df_meta = df_meta[df_meta['Valid QC'].astype(str).str.strip().str.upper() == 'T']
    return df_meta

def get_tau_closing_at_0mV_fixed(data_dict, well_id, feature_prefix):
    tau_values = []
    well_data = data_dict.get(well_id.upper(), {})  # Ensure correct well ID formatting
    if not well_data:
        return np.nan

    # Retrieve values corresponding to v_step (-100 to -40)
    sweep_keys = ['Sweep001', 'Sweep002', 'Sweep003', 'Sweep004']  # Only first 4 sweeps
    for sweep_key in sweep_keys:
        sweep_data = well_data.get(sweep_key, {})
        value = sweep_data.get(feature_prefix, np.nan)
        print(f"Retrieving value for {sweep_key} in well {well_id}, feature {feature_prefix}: {value}")
        tau_values.append(value)

    tau_values = np.array(tau_values, dtype=float)
    print(f"Collected tau values for {well_id}: {tau_values}")

    if len(tau_values) < 4 or np.any(np.isnan(tau_values)):
        return np.nan

    try:
        log_reciprocal = np.log(1 / tau_values)
        slope, intercept, *_ = stats.linregress(v_step[:4], log_reciprocal)
        tau_closing_at_0mV = 1 / np.exp(intercept) * 1000
        print(f"Calculated Tau closing for {well_id}, feature {feature_prefix}: {tau_closing_at_0mV}")
        return tau_closing_at_0mV
    except Exception as e:
        print(f"Error calculating Tau closing for well {well_id}, feature {feature_prefix}: {e}")
        return np.nan

def process_and_save_tail_summary(filepath, data_dict):
    df_meta = extract_meta_columns(filepath)

    if not df_meta.empty:
        df_meta['Tau1_closing_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: get_tau_closing_at_0mV_fixed(data_dict, well_id, 'TailTau1'))
        df_meta['Tau2_closing_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: get_tau_closing_at_0mV_fixed(data_dict, well_id, 'TailTau2'))

        print(df_meta[['Well ID', 'Tau1_closing_at_0mV', 'Tau2_closing_at_0mV']].head())

        output_filename = os.path.splitext(os.path.basename(filepath))[0] + '_Tail_summary.csv'
        df_meta.to_csv(output_filename, index=False)
        print(f"Summary saved to {output_filename}")
    else:
        print(f"No valid data found in {filepath}")

if __name__ == "__main__":
    from Dict_Tail import process_all_files, get_values_across_sweeps

    all_data = process_all_files()
    if all_data:
        for filepath, data_dict in all_data.items():
            process_and_save_tail_summary(filepath, data_dict)
