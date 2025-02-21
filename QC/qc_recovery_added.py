import os
import re
import numpy as np
import pandas as pd
import logging

from qc_function import *  # Added Recovery QC import

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KEYWORDS = ['IV', 'Recovery', 'Tail']

def find_qc_files(folder):
    """Find all CSV files in the given folder that contain any of the keywords."""
    all_files = os.listdir(folder)
    print("All files in folder:", all_files)  # Debugging line
    
    matched_files = {key: [] for key in KEYWORDS}
    for f in all_files:
        clean_filename = f.replace('_', ' ').replace('-', ' ')  
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
    """Process Recovery and Tail QC files and apply both Cell QC and Recovery QC tests."""
    os.makedirs(output_folder, exist_ok=True)
    file_list = find_qc_files(data_folder)
    
    v_steps = np.linspace(-120, 40, 10)
    thresholds = {
        'Seal Resistance': 500,
        'Peak Current': -200,
        'Pre-pulse Leak': 100,
        'Series Resistance': 20,
        'Seal Count': 400,
        'Rundown Lower': 0.7,  # Added Recovery QC thresholds
        'Rundown Upper': 1.3,
        'Peak C1': 300
    }
    
    for key in ['Recovery']:
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
            
            # ✅ **1. Run Cell QC tests**
            cell_qc_results = run_cell_qc_tests(df, v_steps, thresholds)
            
            # ✅ **2. Run Recovery QC tests (New Step Added)**
            recovery_qc_results = run_recovery_qc_tests(df, thresholds)
            
            # ✅ **3. Combine Results into Single DataFrame**
            combined_results = pd.concat([cell_qc_results, recovery_qc_results], axis=1)

            # ✅ **4. Add IV QC reference column**
            combined_results.insert(0, 'IV QC', iv_qc_value)
            
            # ✅ **5. Compute auto_qc including IV QC**
            required_columns = ['IV QC', 'Seal Resistance', 'Seal Count', 'Peak Current', 
                                'Pre-pulse Leak', 'Series Resistance', 'Cap', 'Rundown', 'Peak C1']
            available_columns = [col for col in required_columns if col in combined_results.columns]
            combined_results['Auto QC'] = combined_results[available_columns].prod(axis=1)

            # ✅ **6. Save Output with Modified Filename**
            output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_qc.csv")
            try:
                combined_results.to_csv(output_file, index=False)
                logger.info(f"Processed QC for {file} and saved results to {output_file}")
            except Exception as e:
                logger.error(f"Error saving output file {output_file}: {e}")


if __name__ == "__main__":
    data_folder = "data"
    output_folder = "qc_results"
    iv_qc_folder = "qc_results"  # Assuming IV QC results are stored in the same QC folder
    
    iv_qc_results = get_iv_qc_results(iv_qc_folder)
    process_recovery_tail_qc(data_folder, output_folder, iv_qc_results)
