import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import os
import glob

def recovery_fun(t, a0, a, k):
    return a + (a0 - a) * np.exp(-t / k)

def extract_meta_columns(filepath):
    df = pd.read_csv(filepath, header=[0, 1])
    second_header = df.columns.get_level_values(1).str.strip()
    meta_columns_to_extract = [
        'Experiment  full  Name', 'Experiment Date', 'Well ID', 'Valid QC',
        'Cell Type', 'Cell Concentration', 'Compound Name', 'Concentration',
        'JP Offset (mV)', 'Temperature', 'VMemb (mV)/ IHold (pA)'
    ]

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

# def compute_recovery_params(data_dict, well_id, time_points):
#     well_id_upper = well_id.upper()
#     well_data = data_dict.get(well_id_upper, {})
    
#     if not well_data:
#         print(f"Well ID {well_id_upper} not found in data_dict!")
#         return np.nan, np.nan, np.nan
    
#     sweep_data = well_data.get("Sweep001", {})
#     recovery_values = [sweep_data.get(f'Ratio{i}', np.nan) for i in range(1, 9)]
#     recovery_values = np.array(recovery_values, dtype=float)
    
#     print(recovery_values)
    
#     if np.any(np.isnan(recovery_values)):
#         return np.nan, np.nan, np.nan
    
#     initial_guess = [recovery_values[0], recovery_values[-1], time_points[3]]
#     try:
#         popt, _ = curve_fit(recovery_fun, time_points, recovery_values, p0=initial_guess, maxfev=10000)
#         a0, a, k = popt
#         t50 = -k * np.log((0.5 - a) / (a0 - a))
#     except RuntimeError:
#         print(f"Curve fitting failed for Well ID {well_id_upper}")
#         return np.nan, np.nan, np.nan
#     return round(t50, 2), round(a0, 2), round(a, 2)


def compute_recovery_params(data_dict, well_id, time_points):
    well_id_upper = well_id.upper()
    well_data = data_dict.get(well_id_upper, {})
    
    print(f"Checking well_id: {well_id_upper}")
    
    if not well_data:
        print(f"Well ID {well_id_upper} not found in data_dict!")
        return np.nan, np.nan, np.nan
    

    sweep_data = well_data.get("Sweep001", {})

    recovery_values = [sweep_data.get(f'ratio{i}', np.nan) for i in range(1, 9)]
    
    recovery_values = np.array(recovery_values, dtype=float)
    
    if np.any(np.isnan(recovery_values)):
        return np.nan, np.nan, np.nan
    
    initial_guess = [recovery_values[0], recovery_values[-1], time_points[3]]
    try:
        popt, _ = curve_fit(recovery_fun, time_points, recovery_values, p0=initial_guess, maxfev=10000)
        a0, a, k = popt
        t50 = -k * np.log((0.5 - a) / (a0 - a))
    except RuntimeError:
        print(f"Curve fitting failed for Well ID {well_id_upper}")
        return np.nan, np.nan, np.nan
    
    return round(t50, 2), round(a0, 2), round(a, 2)


def process_and_save_recovery_summary(filepath, data_dict):
    df_meta = extract_meta_columns(filepath)
    
    if df_meta.empty:
        print(f"No valid data found in {filepath}")
        return
       
    time_points = np.array([10, 100, 200, 300, 500, 800, 1200, 1600])
    df_meta['Rec_tau'], df_meta['Rec_A0'], df_meta['Rec_Aend'] = zip(*df_meta['Well ID'].apply(
        lambda well_id: compute_recovery_params(data_dict, well_id, time_points)
    ))

    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '_Recovery_summary.csv'
    df_meta.to_csv(output_filename, index=False)
    print(f"Recovery summary saved to {output_filename}")

if __name__ == "__main__":
    from Dict_Rec import process_all_files  
    
    all_data = process_all_files()
    if all_data:
        for filepath, data_dict in all_data.items():
            process_and_save_recovery_summary(filepath, data_dict)
