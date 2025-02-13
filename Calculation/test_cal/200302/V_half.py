import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy import optimize
import os

def func_boltzmann(x, a, b, c, d):
    return (a + ((b-a)/(1+np.exp((c-x)/d))))

def read_df_iv(file_path, rename_cursor_names=None):
    df_input = pd.read_csv(file_path, header=[0, 1], encoding='utf-8')
    if len(df_input.columns) == 1:
        df_input = pd.read_csv(file_path, sep='\t', skiprows=[2], header=[0, 1], encoding='utf-8')

    df_input = df_input.dropna(how="all").fillna(0)
    df_iv = df_input.copy()

    if rename_cursor_names is not None:
        df_iv = df_iv.rename(columns=rename_cursor_names, level=1)

    column_names_lvl0 = df_iv.columns.levels[0]
    unnamed_columns = [s for s in column_names_lvl0 if "Unnamed: " in s]
    for column in unnamed_columns:
        df_iv = df_iv.rename(columns={column: "Sweep Results"})

    df_iv.columns = pd.MultiIndex.from_tuples(
        [(first.strip(), second.strip()) for first, second in df_iv.columns],
        names=df_iv.columns.names
    )

    Parameter = df_iv.xs('Parameter', level=1, axis=1)
    peak_ac = df_iv.xs('TP1PeakCur', level=1, axis=1)
    peak_in = df_iv.xs('TP2PeakCur', level=1, axis=1)

    return df_iv, Parameter, peak_ac, peak_in

def add_compound_name(output_df, iv_df):
    iv_df_cleaned = iv_df[1:].rename(columns=iv_df.iloc[0])[1:]
    iv_df_cleaned = iv_df_cleaned[['Parameter', 'Compound Name']].iloc[:, :2]  
    merged_df = output_df.merge(iv_df_cleaned, on='Parameter', how='left')
    return merged_df


def process_files(iv_path, output_path):
    df_iv, Parameter, i_peak_ac, i_peak_inac = read_df_iv(iv_path)

    i_peak_ac = np.array(i_peak_ac)
    i_peak_inac = np.array(i_peak_inac)

    min_length = min(len(i_peak_ac), len(i_peak_inac))
    i_peak_ac = i_peak_ac[:min_length]
    i_peak_inac = i_peak_inac[:min_length]

    i_peak_max_ac = i_peak_ac.min() if i_peak_ac.min() != 0 else 0.0000001
    i_peak_max_inac = i_peak_inac.min() if i_peak_inac.min() != 0 else 0.0000001
    i_peak_ac_normalized = i_peak_ac / i_peak_max_ac
    i_peak_inac_normalized = i_peak_inac / i_peak_max_inac

    num_sweeps = len(df_iv.columns.levels[0]) - 1
    v_min = -120
    v_max = v_min + 10 * (num_sweeps - 1)
    v_steps = np.linspace(v_min, v_max, num_sweeps)

    protocol = "Cav3.3"
    gv_curve = {"curve": "linear", "start_voltage": -10, "end_voltage": 30}

    def get_single_v_rev(i_peak_ac_normalized, protocol, gv_curve):
        if np.isnan(i_peak_ac_normalized).any():
            return 10000

        v_input = [0.00000001 if x == 0 else x for x in v_steps]
        ghk_fit = None
        v_rev = None

        if gv_curve["curve"] == "linear":
            index_start = v_steps.tolist().index(gv_curve["start_voltage"])
            index_end = v_steps.tolist().index(gv_curve["end_voltage"]) + 1
            lm = linear_model.LinearRegression()
            lm.fit(
                np.array(np.negative(i_peak_ac_normalized)[index_start:index_end]).reshape(-1, 1),
                v_steps[index_start:index_end]
            )
            v_rev = lm.predict([[0]])[0]
        return v_rev, ghk_fit

    g_ac_normalized_dict = {}
    g_ac_dict = {}
    g_inac_dicts = {}
    v_rev_dict = {}
    ghk_fit_dict = {}
    new_Parameter = []
    for cell_index, wid in enumerate(Parameter.values):
        wid = wid[0]
        i_peak_ac = i_peak_ac_normalized[cell_index]
        i_peak_inac = i_peak_inac_normalized[cell_index]
        v_rev, ghk_fit = get_single_v_rev(i_peak_ac, protocol, gv_curve)
        v_rev_dict[wid] = v_rev
        ghk_fit_dict[wid] = ghk_fit
        if v_rev is None:
            continue
        new_Parameter.append(wid)
        v_diff = v_steps - v_rev
        v_diff = [0.00000001 if x == 0 else x for x in v_diff]
        v_diff = np.array(v_diff[:len(i_peak_ac)])
        g_ac = i_peak_ac / v_diff
        for i in range(len(g_ac) - 3, len(g_ac)):
            fold_change = abs((g_ac[i] - g_ac[i - 1]) / g_ac[i - 1])
            if fold_change > 0.2:
                for j in range(i, len(g_ac)):
                    g_ac[j] = np.nan
                break
        g_max_ac = g_ac[np.isfinite(g_ac)].max()
        if g_max_ac == 0:
            g_max_ac = 0.0000001
        g_ac_normalized_ = g_ac / g_max_ac
        i_peak_max_inac = i_peak_inac.min() if i_peak_inac.min() != 0 else 0.0000001
        g_inac_normalized_ = i_peak_inac / i_peak_max_inac
        g_ac_dict[wid] = g_ac_normalized_
        g_inac_dicts[wid] = g_inac_normalized_

    v_half_ac_list = []
    v_half_inac_list = []
    slope_ac_list = []
    slope_inac_list = []
    cell_index_list = []

    for cell_index in new_Parameter:
        g_ac = g_ac_dict[cell_index]
        v_half_ac = np.nan
        slope_ac = np.nan
        if g_ac is not None:
            is_not_na = np.isfinite(g_ac)
            g_ac_is_not_na = g_ac[is_not_na]
            v_steps_trimmed = v_steps[:len(g_ac_is_not_na)]
            try:
                A_ac, B_ac, v_half_ac, slope_ac = optimize.curve_fit(
                    func_boltzmann,
                    v_steps_trimmed,
                    g_ac_is_not_na,
                    maxfev=5000
                )[0]
            except:
                print(str(cell_index), "Activation: func_boltzmann fitting failed")
        v_half_ac_list.append(v_half_ac)
        slope_ac_list.append(slope_ac)

        g_inac = g_inac_dicts[cell_index]
        v_half_inac = np.nan
        slope_inac = np.nan
        if g_inac is not None:
            is_not_na = np.isfinite(g_inac)
            g_inac_is_not_na = g_inac[is_not_na]
            v_steps_trimmed = v_steps[:len(g_inac_is_not_na)]
            try:
                A_inac, B_inac, v_half_inac, slope_inac = optimize.curve_fit(
                    func_boltzmann,
                    v_steps_trimmed,
                    g_inac_is_not_na,
                    maxfev=5000
                )[0]
            except:
                print(str(cell_index), "Inactivation: func_boltzmann fitting failed")
        v_half_inac_list.append(v_half_inac)
        cell_index_list.append(cell_index)

    output_df = pd.DataFrame({
        "Parameter": cell_index_list,
        "v_half_ac": v_half_ac_list,
        "v_half_inac": v_half_inac_list
    })

    iv_df = pd.read_csv(iv_path)

    result_df_with_compound_name = add_compound_name(output_df, iv_df)
    result_df_with_compound_name.to_csv(output_path, index=False)
    print(f"Processed {iv_path} and saved results to {output_path}")

input_files = ['-75IV_Plate1.csv', '-75IV_Plate2.csv', '-75IV_Plate3.csv', '-100IV_Plate1.csv', '-100IV_Plate2.csv', '-100IV_Plate3.csv']
output_files = ['V_half_75_Plate1.csv', 'V_half_75_Plate2.csv','V_half_75_Plate3.csv', 'V_half_100_Plate1.csv', 'V_half_100_Plate2.csv', 'V_half_100_Plate3.csv']

for iv_path, output_path in zip(input_files, output_files):
    process_files(iv_path, output_path)

print("All files processed.")
