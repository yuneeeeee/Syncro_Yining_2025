import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

v_steps = np.array([-100, -90, -80, -70, -60, -50, -40])

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

def get_tau_closing_at_0mV_fixed(data_dict, well_id, feature_prefix):
    well_id_upper = well_id.upper()
    well_data = data_dict.get(well_id_upper, {})

    if not well_data:
        print(f" Well ID {well_id_upper} not found in data_dict!")
        return np.nan

    tau_values = []
    for i in range(1, 5):
        feature_name = f'{feature_prefix}'.lower()

        sweep_values = [sweep_data.get(feature_name, np.nan) for sweep_data in well_data.values()]

        if sweep_values and not all(np.isnan(sweep_values)):
            tau_values.append(np.nanmean(sweep_values))
        else:
            tau_values.append(np.nan)

    tau_values = np.array(tau_values, dtype=float)
    return np.nan if np.any(np.isnan(tau_values)) else 1 / np.exp(stats.linregress(v_steps[:4], np.log(1 / tau_values))[1]) * 1000


def process_and_save_tail_summary(filepath, data_dict):
    df_meta = extract_meta_columns(filepath)

    if df_meta.empty:
        print(f"No valid data found in {filepath}")
        return


    df_meta['Tau1_closing_at_0mV'] = df_meta['Well ID'].apply(
        lambda well_id: get_tau_closing_at_0mV_fixed(data_dict, well_id, 'TailTau1')
    )
    df_meta['Tau2_closing_at_0mV'] = df_meta['Well ID'].apply(
        lambda well_id: get_tau_closing_at_0mV_fixed(data_dict, well_id, 'TailTau2')
    )
    df_meta[['Tau1_closing_at_0mV', 'Tau2_closing_at_0mV']] = df_meta[['Tau1_closing_at_0mV', 'Tau2_closing_at_0mV']].round(2)

    output_filename = os.path.splitext(os.path.basename(filepath))[0] + '_Tail_summary.csv'
    df_meta.to_csv(output_filename, index=False)
    print(f"Summary saved to {output_filename}")


if __name__ == "__main__":
    from Dict_Tail import process_all_files, get_values_across_sweeps

    all_data = process_all_files()
    if all_data:
        for filepath, data_dict in all_data.items():
            process_and_save_tail_summary(filepath, data_dict)
