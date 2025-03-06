import pandas as pd
import numpy as np
from scipy import stats
import os

input_files = ['-75IV_Plate1.csv', '-75IV_Plate2.csv', '-75IV_Plate3.csv', '-100IV_Plate1.csv', '-100IV_Plate2.csv', '-100IV_Plate3.csv']
output_files = ['tau_inact_75_Plate1.csv', 'tau_inact_75_Plate2.csv','tau_inact_75_Plate3.csv', 'tau_inact_100_Plate1.csv', 'tau_inact_100_Plate2.csv', 'tau_inact_100_Plate3.csv']

def calculate_tau_inact(df):
    v_steps = [-120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    def get_tau_inact_at_0mV(df, input_column, v_steps):
        df_extracted = df.loc[:, [('Sweep Results', 'Parameter')] + [(f'Sweep {i:03d}', input_column) for i in range(1, 18)]]
        df_extracted.columns = ['Parameter'] + [f'Tau ac {i}' for i in range(1, 18)]
        df_extracted_num = df_extracted.apply(pd.to_numeric, errors='coerce')

        tau_inact_at_0mV_list = []
        Parameter_list = []
        for cell_index in df_extracted.index:
            tau_inact_all_sweeps = df_extracted_num.loc[cell_index, 'Tau ac 1':'Tau ac 17'].values
            tau_inact_n20_to_20 = tau_inact_all_sweeps[10:15]
            if np.any(np.isnan(tau_inact_n20_to_20)):
                Parameter_list.append(df_extracted.loc[cell_index, 'Parameter'])
                tau_inact_at_0mV_list.append(np.nan)
                continue
            tau_inact_n20_to_20_log_reciprocal = np.log(1 / tau_inact_n20_to_20)
            slope, intercept, r_value, p_value, std_err = stats.linregress(v_steps[10:15], tau_inact_n20_to_20_log_reciprocal)
            tau_inact_at_0mV = 1000 / np.exp(intercept)
            tau_inact_at_0mV_list.append(tau_inact_at_0mV)
            Parameter_list.append(df_extracted.loc[cell_index, 'Parameter'])

        return Parameter_list, tau_inact_at_0mV_list

    Parameter_list, tau2_result = get_tau_inact_at_0mV(df, 'TP1Tau2In', v_steps)
    Parameter_list, tau1_result = get_tau_inact_at_0mV(df, 'TP1Tau1In', v_steps)

    output_df = pd.DataFrame({
        'Parameter': Parameter_list,
        'TP1Tau1In_at_0mV': tau1_result,
        'TP1Tau2In_at_0mV': tau2_result
    })

    return output_df

def get_compound_name(df):
    df_extracted = df.loc[:, [('Sweep Results', 'Parameter'), ('Sweep 001', 'Compound Name')]]
    df_extracted.columns = ['Parameter', 'Compound Name']
    return df_extracted

def process_files(input_file, output_file):
    df = pd.read_csv(input_file, header=[0, 1])
    df.columns = pd.MultiIndex.from_tuples([(col1.strip(), col2.strip()) for col1, col2 in df.columns])

    compound_name_df = get_compound_name(df) 
    result_df = calculate_tau_inact(df)
    result_df = pd.merge(result_df, compound_name_df, on='Parameter', how='left')

    result_df.to_csv(output_file, index=False)
    print(f"Processed {input_file} and saved results to {output_file}")

# 批处理所有文件
for input_file, output_file in zip(input_files, output_files):
    process_files(input_file, output_file)

print("All files processed.")
