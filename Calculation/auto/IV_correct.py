import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import re
import os
import glob

v_steps = np.array([-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40])

# def extract_meta_columns(filepath):
#     df = pd.read_csv(filepath, header=[0, 1])
#     second_header = df.columns.get_level_values(1).str.strip()
#     meta_columns_to_extract = ['Experiment  full  Name', 'Experiment Date', 'Well ID', 'Valid QC', 'Cell Type', 'Cell Concentration','Compound Name', 'Concentration', 'JP Offset (mV)', 'Temperature', 'VMemb (mV)/ IHold (pA)' ]
#     meta_cols = [col for col, name in zip(df.columns, second_header) if name in meta_columns_to_extract]
#     df_meta = df[meta_cols]
#     df_meta.columns = meta_columns_to_extract
#     if 'Valid QC' in df_meta.columns:
#         df_meta = df_meta[df_meta['Valid QC'].astype(str).str.strip().str.upper() == 'T']
#     return df_meta

def extract_meta_columns(filepath):
    df = pd.read_csv(filepath, header=[0, 1])
    second_header = df.columns.get_level_values(1).str.strip()
    meta_columns_to_extract = [
        'Experiment  full  Name', 'Experiment Date', 'Well ID', 'Valid QC',
        'Cell Type', 'Cell Concentration', 'Compound Name', 'Concentration',
        'JP Offset (mV)', 'Temperature', 'VMemb (mV)/ IHold (pA)'
    ]
    # Extract only the first occurrence of each column
    meta_cols = []
    used_names = set()
    for col, name in zip(df.columns, second_header):
        if name in meta_columns_to_extract and name not in used_names:
            meta_cols.append(col)
            used_names.add(name)

    df_meta = df[meta_cols]
    df_meta.columns = meta_columns_to_extract[:len(meta_cols)]
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

    slope, intercept, _, _, _ = stats.linregress(v_steps[10:15], log_reciprocal)
    return 1000 / np.exp(intercept) if np.isfinite(intercept) else np.nan

# def calculate_v_half_linear(data_dict, well_id, feature):
#     peak_values = get_values_across_sweeps(data_dict, well_id, feature)
#     if len(peak_values) != len(v_step) or len(peak_values) == 0 or all(v == 0 for v in peak_values):
#         return np.nan, np.nan

#     # Normalize using the maximum absolute value
#     peak_max = min(np.array(peak_values)) if min(np.array(peak_values)) != 0 else 1e-7
#     peak_values_normalized = np.array(peak_values) / peak_max

#     try:
#         model = LinearRegression()
#         model.fit(np.negative(peak_values_normalized).reshape(-1, 1), v_step)
#         v_half = model.predict([[0]])[0]
#         slope = model.coef_[0]
#         return v_half, slope
#     except Exception as e:
#         print(f"Linear regression failed for well {well_id} and feature {feature}: {e}")
#         return np.nan, np.nan
def boltzmann_func(v, A, B, v_half, slope):
    """Boltzmann function used to fit activation and inactivation curves."""
    return A + (B - A) / (1 + np.exp((v_half - v) / slope))

def calculate_v_rev(i_peak_ac, v_steps):
    """Calculate reversal potential (V_rev) using linear regression in the range -10 to 30 mV."""
    if np.isnan(i_peak_ac).any():
        return np.nan
    
    gv_curve = {"curve": "linear", "start_voltage": -10, "end_voltage": 30}
    
    index_start = v_steps.tolist().index(gv_curve["start_voltage"])
    index_end = v_steps.tolist().index(gv_curve["end_voltage"]) + 1
    
    valid_indices = ~np.isnan(i_peak_ac)
    i_peak_ac_valid = i_peak_ac[index_start:index_end][valid_indices[index_start:index_end]]
    v_steps_valid = v_steps[index_start:index_end][valid_indices[index_start:index_end]]
    
    if len(i_peak_ac_valid) < 2:
        return np.nan
    
    model = LinearRegression()
    model.fit(np.negative(i_peak_ac_valid).reshape(-1, 1), v_steps_valid)
    v_rev = model.predict([[0]])[0]
    
    return v_rev

def calculate_conductance(i_peak, v_steps, v_rev):
    """Compute conductance G(V) with artifact filtering."""
    v_diff = v_steps - v_rev
    v_diff[v_diff == 0] = 1e-7  # Avoid division by zero
    g_values = i_peak / v_diff
    
    for i in range(len(g_values) - 3, len(g_values)):
        if abs((g_values[i] - g_values[i - 1]) / g_values[i - 1]) > 0.2:
            g_values[i:] = np.nan
            break
    
    g_max = np.nanmax(g_values)
    g_normalized = g_values / g_max if g_max != 0 else g_values
    
    return g_normalized

def fit_boltzmann(v_steps, g_normalized):
    """Fit data to the Boltzmann equation to estimate V_half and slope."""
    valid_indices = ~np.isnan(g_normalized)
    v_fit = v_steps[valid_indices]
    g_fit = g_normalized[valid_indices]
    
    if len(g_fit) < 4:
        return np.nan, np.nan
    
    try:
        popt, _ = curve_fit(boltzmann_func, v_fit, g_fit, maxfev=5000)
        v_half, slope = popt[2], popt[3]
        return v_half, slope
    except:
        return np.nan, np.nan

def calculate_v_half(data_dict, well_id, v_steps, feature_ac, feature_inac):
    """Calculate V_rev, V_half activation, slope activation, V_half inactivation, and slope inactivation."""
    i_peak_ac = np.array(get_values_across_sweeps(data_dict, well_id, feature_ac))
    i_peak_inac = np.array(get_values_across_sweeps(data_dict, well_id, feature_inac))

    v_rev = calculate_v_rev(i_peak_ac, v_steps)

    if np.isnan(v_rev):
        return np.nan, np.nan, np.nan, np.nan, np.nan  # Return NaNs if V_rev calculation fails

    g_ac_normalized = calculate_conductance(i_peak_ac, v_steps, v_rev)
    i_peak_max_inac = i_peak_inac.min() if i_peak_inac.min() != 0 else 1e-7
    g_inac_normalized = i_peak_inac / i_peak_max_inac

    v_half_ac, slope_ac = fit_boltzmann(v_steps, g_ac_normalized)
    v_half_inac, slope_inac = fit_boltzmann(v_steps, g_inac_normalized)

    return v_rev, v_half_ac, slope_ac, v_half_inac, slope_inac



def process_and_save_summary(filepath, data_dict):
    df_meta = extract_meta_columns(filepath)
    df_meta['TP1_Max_CurDen'] = df_meta['Well ID'].apply(lambda well_id: calculate_max_cd_ac(data_dict, well_id, 'TP1CurDen'))
    df_meta['TP2_Max_CurDen'] = df_meta['Well ID'].apply(lambda well_id: calculate_max_cd_ac(data_dict, well_id, 'TP2CurDen'))
    df_meta['TP1Tau1In_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: calculate_tau_inact(data_dict, well_id, 'TP1Tau1In'))
    df_meta['TP1Tau2In_at_0mV'] = df_meta['Well ID'].apply(lambda well_id: calculate_tau_inact(data_dict, well_id, 'TP1Tau2In'))
    df_meta[['V_rev', 'V_half_ac', 'Slope_ac', 'V_half_inac', 'Slope_inac']] = df_meta['Well ID'].apply(
        lambda well_id: pd.Series(calculate_v_half(data_dict, well_id, v_steps, 'TP1PeakCur', 'TP2PeakCur'))
    )

    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '_summary.csv'
    df_meta[['TP1_Max_CurDen', 'TP2_Max_CurDen', 'TP1Tau1In_at_0mV', 'TP1Tau2In_at_0mV', 'V_rev', 'V_half_ac', 'Slope_ac', 'V_half_inac', 'Slope_inac']] = df_meta[['TP1_Max_CurDen', 'TP2_Max_CurDen', 'TP1Tau1In_at_0mV', 'TP1Tau2In_at_0mV', 'V_rev', 'V_half_ac', 'Slope_ac', 'V_half_inac', 'Slope_inac']].round(2)
    df_meta.to_csv(output_filename, index=False)
    print(f"Summary saved to {output_filename}")

if __name__ == "__main__":
    from Dict_IV import process_all_files, get_values_across_sweeps

    all_data = process_all_files()
    if all_data:
        for filepath, data_dict in all_data.items():
            process_and_save_summary(filepath, data_dict)
