import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def test_manual_qc(df, cell_index):
    valid_qc_column = df.columns.get_level_values(1) == 'Valid QC'
    if valid_qc_column.any():
        value = df.loc[cell_index, valid_qc_column].values[0]
        return value == 'T', value
    return True, None

def get_i_peak_max_sweep(df, v_steps):
    columns_with_i_peak_max_ac = df.xs('TP1PeakCur', level=1, axis=1).idxmin(axis=1)
    columns_with_i_peak_max_ac[columns_with_i_peak_max_ac.isna()] = 'Sweep 010'

    output_object = {
        'i_peak_max_sweep_num': columns_with_i_peak_max_ac,
    }

    logger.debug('Finished getting i_peak_max_sweep_num')
    return output_object


def test_iv_jump(df, cell_index, threshold):
    i_peak_list = pd.to_numeric(df.xs('TP1PeakCur', level=1, axis=1).iloc[cell_index], errors='coerce')
    i_peak_max = i_peak_list.min()
    i_peak_normalized_list = i_peak_list / i_peak_max
    for index in range(1, len(i_peak_normalized_list)):
        value = abs(i_peak_normalized_list[index] - i_peak_normalized_list[index-1])
        if value >= threshold:
            return False, value
    return True, value

def test_seal_resistance(current_cell, threshold):
    seal_r = pd.to_numeric(current_cell.xs("Seal Resistance", level=1, axis=0), errors='coerce').mean(axis=0)
    value = seal_r / 1e6
    return value > threshold, value

def test_seal_count(current_cell, threshold):
    cell = pd.to_numeric(current_cell.xs("Seal Resistance", level=1, axis=0), errors='coerce')
    data = cell / 1e6
    fail_count = (data < threshold).sum()
    pass_flag = fail_count < 2
    return pass_flag, fail_count

def test_peak_current(df, cell_index, threshold):
    i_peak_list = pd.to_numeric(df.xs('TP1PeakCur', level=1, axis=1).iloc[cell_index], errors='coerce')
    min_peak = i_peak_list.min()
    value = min_peak * 1e12
    return value < threshold, value

def test_pre_pulse_leak(df, cell_index, threshold):
    pre_pulse = abs(pd.to_numeric(df['Sweep 001'][' Leak prep'][cell_index], errors='coerce'))
    value = pre_pulse * 1e12
    return value < threshold, value

def test_leak_steady(df, cell_index, v_steps, threshold):
    i_peak_max_sweep_num_list = get_i_peak_max_sweep(df, v_steps)['i_peak_max_sweep_num']
    i_peak_max_sweep_num = i_peak_max_sweep_num_list[cell_index]
    steady_state_leak = abs(pd.to_numeric(df[i_peak_max_sweep_num]['Leak stead'][cell_index], errors='coerce'))
    I_peak = abs(pd.to_numeric(df[i_peak_max_sweep_num]['TP1PeakCur'][cell_index], errors='coerce'))
    qc = steady_state_leak/I_peak
    return steady_state_leak < (I_peak * threshold), qc


def test_series_resistance(current_cell, threshold):
    ser_r = pd.to_numeric(current_cell.xs("Series Resistance", level=1, axis=0), errors='coerce').mean(axis=0)
    value = ser_r / 1e6
    return value < threshold, value

def test_cap_mean(current_cell, lower_bound=4.5, upper_bound=50):
    cap = pd.to_numeric(current_cell.xs("Capacitance", level=1, axis=0), errors='coerce').mean(axis=0)
    cap_values = cap * 1e12
    if lower_bound <= cap_values <= upper_bound:
        return True, cap_values
    else:
        return False, cap_values
    
def test_rundown(current_cell, lower_bound=0.7, upper_bound=1.3):
    rundown = pd.to_numeric(current_cell.xs(" Rundown", level=1, axis=0), errors='coerce').iloc[0]
    return lower_bound <= rundown <= upper_bound, rundown




def test_peak_c1(df, cell_index, threshold):
    i_peak_list = pd.to_numeric(df.xs(' C 1', level=1, axis=1).iloc[cell_index], errors='coerce')
    min_peak = i_peak_list.min()
    value = abs(min_peak) * 1e12  # Convert to pA
    return value >= threshold, value

cell_qc_tests = {
    'Seal Resistance': 'test_seal_resistance',
    'Seal Count': 'test_seal_count',
    'Peak Current': 'test_peak_current',
    'Pre-pulse Leak': 'test_pre_pulse_leak',
    'Series Resistance': 'test_series_resistance',
    'Cap': 'test_cap_mean'
}

iv_qc_tests = {
    'IV Jump': 'test_iv_jump',
    'Peak Current': 'test_peak_current',
    'Leak Steady': 'test_leak_steady'
}

recovery_qc_tests = {
    "Rundown": "test_rundown",
    "Peak C1": "test_peak_c1"
}

def run_cell_qc_tests(df, v_steps, thresholds):
    results = []
    for cell_index in range(len(df)):
        current_cell = df.iloc[cell_index]
        
        seal_resistance_result, seal_resistance_value = test_seal_resistance(current_cell, thresholds['Seal Resistance'])
        seal_count_result, seal_count_value = test_seal_count(current_cell, thresholds['Seal Resistance'])
        pre_pulse_leak_result, pre_pulse_leak_value = test_pre_pulse_leak(df, cell_index, thresholds['Pre-pulse Leak'])
        # leak_steady_result, leak_steady_value = test_leak_steady(df, cell_index, v_steps, thresholds['Leak Steady'])
        series_resistance_result, series_resistance_value = test_series_resistance(current_cell, thresholds['Series Resistance'])
        cap_result, cap_value = test_cap_mean(current_cell)

        test_result = {
            'Seal Resistance': int(seal_resistance_result),
            'Seal Count': int(seal_count_result),
            'Pre-pulse Leak': int(pre_pulse_leak_result),
            'Series Resistance': int(series_resistance_result),
            'Cap': int(cap_result),
            'Seal Resistance Value': seal_resistance_value,
            'Seal Count Value': seal_count_value,
            'Pre-pulse Leak Value': pre_pulse_leak_value,
            'Series Resistance Value': series_resistance_value,
            'Cap Value': cap_value
        }
        results.append(test_result)
    return pd.DataFrame(results)

def run_iv_qc_tests(df, v_steps, thresholds):
    results = []
    for cell_index in range(len(df)):
        iv_jump_result, iv_jump_value = test_iv_jump(df, cell_index, thresholds['IV Jump'])
        peak_current_result, peak_current_value = test_peak_current(df, cell_index, thresholds['Peak Current'])
        leak_steady_result, leak_steady_value = test_leak_steady(df, cell_index, v_steps, thresholds['Leak Steady'])
        
        test_result = {
            'IV Jump': int(iv_jump_result),
            'Peak Current': int(peak_current_result),
            'Leak Steady': int(leak_steady_result),
            'IV Jump Value': iv_jump_value,
            'Peak Current Value': peak_current_value,
            'Leak Steady Value': leak_steady_value
            
        }
        results.append(test_result)
    return pd.DataFrame(results)

def run_recovery_qc_tests(df, thresholds):

    results = []
    for cell_index in range(len(df)):
        current_cell = df.iloc[cell_index]

        # Run Rundown test
        rundown_result, rundown_value = test_rundown(current_cell, 
                                                     lower_bound=thresholds.get("Rundown Lower", 0.8), 
                                                     upper_bound=thresholds.get("Rundown Upper", 1.2))

        # Run Peak C1 test
        peak_c1_result, peak_c1_value = test_peak_c1(df, cell_index, threshold=thresholds.get("Peak C1", 300))

        # Store results
        test_result = {
            'Rundown': int(rundown_result),
            'Peak C1': int(peak_c1_result),
            'Rundown Value': rundown_value,
            'Peak C1 Value': peak_c1_value
        }
        results.append(test_result)

    return pd.DataFrame(results)


# def main(input_file, output_file,final_output_file):

#     df = pd.read_csv(input_file, header=[0, 1])
#     v_steps = np.linspace(-120, 40, 10)  # Example voltage steps
#     thresholds = {
#         'IV Jump': 0.7,
#         'Seal Resistance': 500,
#         'Peak Current': -200,
#         'Pre-pulse Leak': 100,
#         'Leak Steady': 0.15,
#         'Series Resistance': 20,
#         'Seal Count': 400
#     }


#     well_ids = pd.read_csv(input_file).iloc[1:, 0].reset_index(drop=True)
    
#     results_df = run_tests(df, v_steps, thresholds)
#     results_df.to_csv(output_file, index=False)
    
#     final_df = pd.read_csv(output_file)
#     final_df = final_df.drop(index=0)  
#     final_df['well_id'] = well_ids
#     final_df.to_csv(final_output_file, index=False)
    
#     return final_output_file


# inlist = ['test_IV.CSV']
# out_first=['test.csv']
# outlist = ['_final.csv']

# for (input_file, output_file,output_file_final) in zip(inlist, out_first,outlist):
#     output_file_path = main(input_file, output_file,output_file_final)
#     output_file_path
    
# # output_file_path = main()
# # output_file_path
