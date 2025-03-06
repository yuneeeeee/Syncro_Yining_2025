import pandas as pd
import numpy as np
from scipy import stats
import re
import os
import glob

v_step = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]  # Store v_step globally for later calculations

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

def calculate_max_cd_ac(data_dict, well_id, feature):
    values = [abs(v) for v in get_values_across_sweeps(data_dict, well_id, feature) if v is not None]
    return -max(values) if values else None

# def calculate_v_step_from_dict(data_dict):
#     global v_step
#     first_well = next(iter(data_dict.values()), {})
#     v_step = [float(values.get('Voltage', 0)) * 1000 for values in first_well.values() if 'Voltage' in values]

def calculate_tau_inact(data_dict, well_id, feature):
    tau_inact_values = get_values_across_sweeps(data_dict, well_id, feature)
    tau_inact_n20_to_20 = tau_inact_values[10:15]

    if len(tau_inact_n20_to_20) < 5 or any(np.isnan(tau_inact_n20_to_20)):
        return np.nan

    log_reciprocal = np.log(np.where(tau_inact_n20_to_20 != 0, 1 / np.array(tau_inact_n20_to_20), np.nan))
    if np.any(np.isnan(log_reciprocal)):
        return np.nan

    slope, intercept, _, _, _ = stats.linregress(v_step[10:15], log_reciprocal)
    return 1000 / np.exp(intercept) if np.isfinite(intercept) else np.nan

def process_and_save_summary(filepath, data_dict):
    df_meta = extract_meta_columns(filepath)
    df_meta['TP1_Max_CurDen'] = df_meta['Well ID'].apply(lambda well_id: calculate_max_cd_ac(data_dict, well_id, 'TP1CurDen'))
    df_meta['TP2_Max_CurDen'] = df_meta['Well ID'].apply(lambda well_id: calculate_max_cd_ac(data_dict, well_id, 'TP2CurDen'))
    df_meta['TP1Tau1In_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: calculate_tau_inact(data_dict, well_id, 'TP1Tau1In'))
    df_meta['TP1Tau2In_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: calculate_tau_inact(data_dict, well_id, 'TP1Tau2In'))
    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '_summary.csv'
    df_meta.to_csv(output_filename, index=False)
    print(f"Summary saved to {output_filename}")

if __name__ == "__main__":
    from Dict_IV import process_all_files, get_values_across_sweeps

    all_data = process_all_files()
    if all_data:
        for filepath, data_dict in all_data.items():
            process_and_save_summary(filepath, data_dict)
