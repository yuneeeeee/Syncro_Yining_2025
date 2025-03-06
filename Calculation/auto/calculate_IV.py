# Updated calculate_IV.py to fix TypeError in get_sweep_range
import pandas as pd
import numpy as np
from scipy import stats
from scipy import optimize
import os
import glob
import re

def get_sweep_range(df, column_pattern=r"Sweep \\d+"):
    sweep_columns = [str(col) for col in df.columns if isinstance(col, (str, tuple)) and re.match(column_pattern, str(col))]
    sweep_indices = sorted(set(int(re.search(r"\\d+", col).group()) for col in sweep_columns if re.search(r"\\d+", col)))
    return sweep_indices

def filter_valid_qc(df):
    if isinstance(df.columns, pd.MultiIndex):
        valid_qc_col = next((col for col in df.columns if col[1].strip().lower() == 'valid qc'), None)
        if valid_qc_col:
            return df.loc[df[valid_qc_col].astype(str).str.strip().str.upper().eq('T')].copy()  # 'T' means pass
    print("Warning: 'Valid QC' column not found. Using all wells.")
    return df.copy()

def calculate_max_cd(df):
    sweep_indices = get_sweep_range(df)
    df_extracted = df[[f'Sweep {i:03d} TP1CurDen' for i in sweep_indices] + [f'Sweep {i:03d} TP2CurDen' for i in sweep_indices]].copy()

    def get_largest_absolute(df, prefix):
        cols = [col for col in df.columns if prefix in col]
        return df[cols].apply(lambda row: row.loc[row.abs().idxmax()] if not row.isna().all() else np.nan, axis=1)

    df_extracted['TP1_Max_CurDen'] = get_largest_absolute(df_extracted, 'TP1CurDen')
    df_extracted['TP2_Max_CurDen'] = get_largest_absolute(df_extracted, 'TP2CurDen')
    return df_extracted[['TP1_Max_CurDen', 'TP2_Max_CurDen']]

def calculate_tau_inact(df):
    v_steps = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
    sweep_indices = get_sweep_range(df)

    def get_tau_inact(df, col_name):
        cols = [f'Sweep {i:03d} {col_name}' for i in sweep_indices]
        df_extracted = df[cols].copy()

        results = []
        for _, row in df_extracted.iterrows():
            tau_subset = row.iloc[11:16].dropna()
            if len(tau_subset) < 5:
                results.append(np.nan)
            else:
                slope, intercept, *_ = stats.linregress(v_steps[10:15], np.log(1 / tau_subset))
                results.append(1000 / np.exp(intercept))
        return results

    return pd.DataFrame({
        'TP1Tau1In_at_0mV': get_tau_inact(df, 'TP1Tau1In'),
        'TP1Tau2In_at_0mV': get_tau_inact(df, 'TP1Tau2In')
    })

def calculate_v_half(df):
    v_steps = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]
    sweep_indices = get_sweep_range(df)

    def boltzmann(x, a, b, c, d):
        return a + (b - a) / (1 + np.exp((c - x) / d))

    def fit_boltzmann(values):
        try:
            popt, _ = optimize.curve_fit(boltzmann, v_steps, values, maxfev=5000)
            return popt[2], popt[3]
        except Exception:
            return np.nan, np.nan

    df_extracted = df[[f'Sweep {i:03d} TP1PeakCur' for i in sweep_indices]].copy()

    v_half_values, slope_values = [], []
    for _, row in df_extracted.iterrows():
        g_values = row.dropna().values
        v_half, slope = fit_boltzmann(g_values) if len(g_values) == len(v_steps) else (np.nan, np.nan)
        v_half_values.append(v_half)
        slope_values.append(slope)

    return pd.DataFrame({'V_half': v_half_values, 'Slope': slope_values})

def get_compound_name(df):
    compound_name_col = next((col for col in df.columns if isinstance(col, str) and 'compound name' in col.lower()), None)
    if compound_name_col:
        return df[[compound_name_col]].rename(columns={compound_name_col: 'Compound Name'})
    else:
        return pd.DataFrame({'Compound Name': [np.nan] * len(df)})

def process_file(filepath):
    df = pd.read_csv(filepath, header=[0, 1])  # Read with MultiIndex header
    df_filtered = filter_valid_qc(df)
    if df_filtered.empty:
        print(f"No valid rows found in {filepath}. Skipping.")
        return

    max_cd = calculate_max_cd(df_filtered)
    tau_inact = calculate_tau_inact(df_filtered)
    v_half = calculate_v_half(df_filtered)
    compound_names = get_compound_name(df_filtered)

    result = pd.concat([max_cd.reset_index(drop=True), tau_inact.reset_index(drop=True), v_half.reset_index(drop=True), compound_names.reset_index(drop=True)], axis=1)
    output_path = os.path.join(os.path.dirname(filepath), 'results', os.path.basename(filepath).replace('.csv', '_calculation_result.csv'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Processed {filepath} and saved as {output_path}")

for file in glob.glob("*IV*.csv"):
    process_file(file)

print("All files processed.")
