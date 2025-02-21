import os
import pandas as pd

def batch_process_files(folder_path):
    """
    Batch process all CSV files in the folder:
    - If 'Auto QC' column == 1, add 'Valid QC' = 'T'
    - If 'Auto QC' column == 0, add 'Valid QC' = 'F'
    - Save each modified file, overwriting the original.
    """
    # Get list of all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if 'Auto QC' column exists
            if 'Auto QC' in df.columns:
                # Add 'Valid QC' column based on 'Auto QC' value
                df['Valid QC'] = df['Auto QC'].apply(lambda x: 'T' if x == 1 else 'F')
                
                # Save the modified file (overwrite original)
                df.to_csv(file_path, index=False)
                print(f"Processed and saved: {file_name}")
            else:
                print(f"Skipped (no 'Auto QC' column): {file_name}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Example: Process all files in the current directory
batch_process_files('.')

