import os
import re
import numpy as np
import pandas as pd
import logging

from qc_function import run_cell_qc_tests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KEYWORDS = ['IV', 'Recovery', 'Tail']

def find_qc_files(folder):
    """Find all CSV files in the given folder that contain any of the keywords."""
    all_files = os.listdir(folder)
    print("All files in folder:", all_files)  # Debugging line
    
    matched_files = {key: [] for key in KEYWORDS}
    for f in all_files:
        clean_filename = f.replace('_', ' ').replace('-', ' ')  # 
        for key in KEYWORDS:
            if re.search(rf'(?i)\b{key}\b', clean_filename) and f.lower().endswith('.csv'):
                matched_files[key].append(f)
    
    print("Matched files:", matched_files)  # Debugging line
    return matched_files


def get_iv_qc_results(qc_results_folder):
    """Load IV QC results and store them for reference."""
    iv_qc_results = {}
    for file in os.listdir(qc_results_folder):
        if 'IV_qc.csv' in file:
            df = pd.read_csv(os.path.join(qc_results_folder, file))
            if 'Auto QC' in df.columns:
                iv_qc_results[file.replace('_qc.csv', '')] = df['Auto QC'].values
    return iv_qc_results

def process_recovery_tail_qc(data_folder, output_folder, iv_qc_results):
    os.makedirs(output_folder, exist_ok=True)
    file_list = find_qc_files(data_folder)
    
    v_steps = np.linspace(-120, 40, 10)
    thresholds = {
        'Seal Resistance': 500,
        'Peak Current': -200,
        'Pre-pulse Leak': 100,
        'Leak Steady': 0.15,
        'Series Resistance': 20,
        'Seal Count': 400
    }
    
    for key in ['Recovery', 'Tail']:
        for file in file_list[key]:
            file_path = os.path.join(data_folder, file)
            try:
                df = pd.read_csv(file_path, header=[0, 1])
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue
            
            logger.info(f"Processing file: {file}")
            
            # Find corresponding IV file's Auto QC value
            iv_filename = re.sub(rf'(?i){key}', 'IV', file)
            iv_qc_value = iv_qc_results.get(iv_filename, [0] * len(df))
            
            # Run Cell QC tests
            cell_qc_results = run_cell_qc_tests(df, v_steps, thresholds)
            
            # Add IV QC reference column
            cell_qc_results.insert(0, 'IV QC', iv_qc_value)
            
            # Compute auto_qc including IV QC
            required_columns = ['IV QC', 'Seal Resistance', 'Seal Count', 'Peak Current', 'Pre-pulse Leak', 'Series Resistance', 'Cap']
            available_columns = [col for col in required_columns if col in cell_qc_results.columns]
            cell_qc_results['Auto QC'] = cell_qc_results[available_columns].prod(axis=1)
            
            # Save output with modified filename
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_qc.csv")
            try:
                cell_qc_results.to_csv(output_file, index=False)
                logger.info(f"Processed QC for {file} and saved results to {output_file}")
            except Exception as e:
                logger.error(f"Error saving output file {output_file}: {e}")
    
if __name__ == "__main__":
    data_folder = "data"
    output_folder = "qc_results"
    iv_qc_folder = "qc_results"  # Assuming IV QC results are stored in the same QC folder
    iv_qc_results = get_iv_qc_results(iv_qc_folder)
    process_recovery_tail_qc(data_folder, output_folder, iv_qc_results)
