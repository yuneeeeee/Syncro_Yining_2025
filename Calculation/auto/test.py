import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import re
import os
import glob

v_step = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]

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

def calculate_v_half_linear(data_dict, well_id, feature):
    peak_values = get_values_across_sweeps(data_dict, well_id, feature)

    if len(peak_values) != len(v_step) or len(peak_values) == 0 or all(v == 0 for v in peak_values):
        return np.nan, np.nan

    peak_min = abs(min(peak_values)) if min(peak_values) != 0 else 1e-7
    peak_values_normalized = np.array(peak_values) / peak_min

    if feature == 'TP2PeakCur':
        peak_values_normalized = -peak_values_normalized  # Ensure correct sign for inactivation

    try:
        model = LinearRegression()
        model.fit(np.negative(peak_values_normalized).reshape(-1, 1), v_step)
        v_half = model.predict([[0]])[0]
        slope = model.coef_[0]
        return v_half, slope
    except Exception as e:
        print(f"Linear regression failed for well {well_id} and feature {feature}: {e}")
        return np.nan, np.nan

def process_and_save_summary(filepath, data_dict):
    df_meta = extract_meta_columns(filepath)
    df_meta['TP1_Max_CurDen'] = df_meta['Well ID'].apply(lambda well_id: calculate_max_cd_ac(data_dict, well_id, 'TP1CurDen'))
    df_meta['TP2_Max_CurDen'] = df_meta['Well ID'].apply(lambda well_id: calculate_max_cd_ac(data_dict, well_id, 'TP2CurDen'))
    df_meta['TP1Tau1In_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: calculate_tau_inact(data_dict, well_id, 'TP1Tau1In'))
    df_meta['TP1Tau2In_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: calculate_tau_inact(data_dict, well_id, 'TP1Tau2In'))
    df_meta[['V_half_ac', 'Slope1']] = df_meta['Well ID'].apply(lambda well_id: pd.Series(calculate_v_half_linear(data_dict, well_id, 'TP1PeakCur')))
    df_meta[['V_half_inac', 'Slope2']] = df_meta['Well ID'].apply(lambda well_id: pd.Series(calculate_v_half_linear(data_dict, well_id, 'TP2PeakCur')))

    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '_summary.csv'
    df_meta.to_csv(output_filename, index=False)
    print(f"Summary saved to {output_filename}")

if __name__ == "__main__":
    from Dict_IV import process_all_files, get_values_across_sweeps

    all_data = process_all_files()
    if all_data:
        for filepath, data_dict in all_data.items():
            process_and_save_summary(filepath, data_dict)
