import os
import pandas as pd
import numpy as np

def get_largest_absolute(df, keyword):
    """Returns the max absolute value for a given metric across all sweeps."""
    relevant_cols = [col for col in df.columns if keyword in col[1].replace(" ", "")]
    
    if not relevant_cols:
        print(f"No columns found for {keyword}.")
        return None
    
    return df[relevant_cols].apply(lambda row: row.abs().max(), axis=1)

def calculate_max_cd(df):
    """Calculates TP1_Max_CurDen and TP2_Max_CurDen dynamically for each well."""
    
    # Identify well ID column
    well_id_col = next((col for col in df.columns if col[1].strip() == 'Well ID'), None)
    if not well_id_col:
        print("'Well ID' column not found.")
        return None
    
    # Dynamically extract max absolute TP1CurDen and TP2CurDen across all sweeps
    df['TP1_Max_CurDen'] = get_largest_absolute(df, 'TP1CurDen')
    df['TP2_Max_CurDen'] = get_largest_absolute(df, 'TP2CurDen')

    # DEBUG: Print column names to check if they exist
    print("Columns in DataFrame after calculation:", df.columns)

    # Ensure the new columns are properly assigned
    if 'TP1_Max_CurDen' not in df.columns or 'TP2_Max_CurDen' not in df.columns:
        print("Error: Max CurDen columns were not added correctly!")
        return None

    # Return only filtered Well ID and calculated results
    return df[[well_id_col] + ['TP1_Max_CurDen', 'TP2_Max_CurDen']]

def process_iv_calculations(file_path):
    """Preprocesses and filters the IV file before calculations and saves a clean summary file."""
    
    # Read the CSV with multi-line headers
    df = pd.read_csv(file_path, header=[0, 1])
    
    # Clean up column headers
    df.columns = pd.MultiIndex.from_tuples([(col1.strip(), col2.strip()) for col1, col2 in df.columns])
    
    # Identify Valid QC column
    valid_qc_col = next((col for col in df.columns if col[1].strip() == 'Valid QC'), None)
    if not valid_qc_col:
        print(f"'Valid QC' column not found in {file_path}.")
        return
    
    # Filter out rows where 'Valid QC' is 'F' (case-insensitive, trimming spaces)
    df = df[~df[valid_qc_col].astype(str).str.strip().str.upper().eq('F')]
    
    # Perform IV calculations
    summary_df = calculate_max_cd(df)
    
    if summary_df is not None:
        # Generate new summary file name
        base_name = os.path.basename(file_path).replace(".CSV", "")
        summary_file = os.path.join(os.path.dirname(file_path), f"{base_name}_summary.CSV")
        
        # Save the summary file with only filtered Well IDs and calculation results
        summary_df.to_csv(summary_file, index=False)
        print(f"Processed IV calculations saved as a new file: {summary_file}")
