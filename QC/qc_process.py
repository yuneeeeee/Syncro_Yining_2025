import os
import numpy as np
import pandas as pd
import logging
import re

from qc_function import run_cell_qc_tests, run_iv_qc_tests, test_manual_qc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def find_iv_files(folder):
#     return [f for f in os.listdir(folder) if 'IV' in f.replace(' ', '').upper() and f.endswith('.csv')]



def find_iv_files(folder):
    """Find all CSV files in the given folder that contain 'IV' (case-insensitive, ignoring spaces and special characters) in their filename."""
    all_files = os.listdir(folder)
    print("All files in folder:", all_files)  # Debugging line
    matched_files = [f for f in all_files if re.search(r'(?i)\bIV\b', f.replace('_', ' ').replace('-', ' ')) and f.endswith('.CSV')]
    print("Matched files:", matched_files)  # Debugging line
    return matched_files

def process_qc_files(data_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    file_list = find_iv_files(data_folder)
    
    if not file_list:
        logger.warning("No IV files found in the specified data folder.")
        return
    
    v_steps = np.linspace(-120, 40, 10)
    thresholds = {
        'IV Jump': 0.7,
        'Seal Resistance': 500,
        'Peak Current': -200,
        'Pre-pulse Leak': 100,
        'Leak Steady': 0.15,
        'Series Resistance': 20,
        'Seal Count': 400
    }
    
    for file in file_list:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path, header=[0, 1])
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue
        
        logger.info(f"Processing file: {file}")
        
        # Extract Well ID column
        well_id_column = df.columns.get_level_values(1) == 'Well ID'
        well_ids = df.loc[:, well_id_column].values.flatten() if well_id_column.any() else None
        
        # Run QC tests
        cell_qc_results = run_cell_qc_tests(df, v_steps, thresholds)
        iv_qc_results = run_iv_qc_tests(df, v_steps, thresholds)
        
        # Retrieve manual QC results as reference
        manual_qc_results = []
        for cell_index in range(len(df)):
            manual_qc_result, manual_qc_value = test_manual_qc(df, cell_index)
            manual_qc_results.append({'Manual QC': int(manual_qc_result), 'Manual QC Value': manual_qc_value})
        manual_qc_df = pd.DataFrame(manual_qc_results)
        
        # Combine all results
        final_results = pd.concat([manual_qc_df, iv_qc_results, cell_qc_results], axis=1)
        
        # Compute auto_qc using only specified columns
        required_columns = ['IV Jump', 'Peak Current', 'Seal Resistance', 'Seal Count', 'Pre-pulse Leak', 'Leak Steady', 'Series Resistance', 'Cap']
        available_columns = [col for col in required_columns if col in final_results.columns]
        final_results['Auto QC'] = final_results[available_columns].prod(axis=1)
        
        # Add match column: 1 if Auto QC matches Manual QC, else 0
        final_results['Match'] = (final_results['Auto QC'] == final_results['Manual QC']).astype(int)
        
        # Add Well ID column if available
        if well_ids is not None:
            final_results.insert(0, 'Well ID', well_ids)
        
        # Save output with modified filename
        output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_qc.csv")
        try:
            final_results.to_csv(output_file, index=False)
            logger.info(f"Processed QC for {file} and saved results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving output file {output_file}: {e}")
    
if __name__ == "__main__":
    data_folder = "data"
    output_folder = "qc_results"
    process_qc_files(data_folder, output_folder)
