import os
import pandas as pd
import io

def batch_replace_valid_qc(folder_path):
    """
    For each pair of files in the folder with '_qc' and '_auto' in their names:
    - Load both files.
    - Replace 'Valid QC' in the '_auto' file with 'Valid QC' from the '_qc' file.
    - Keep the original structure of the '_auto' file (no row or header changes).
    - Ensure no extra blank lines appear in the output.
    """
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Extract files with '_qc' and '_auto' in their names
    qc_files = [f for f in files if '_qc' in f and f.lower().endswith('.csv')]
    auto_files = [f for f in files if '_auto' in f and f.lower().endswith('.csv')]
    
    # Create dictionaries to map filenames without "_qc" or "_auto" to full filenames
    qc_map = {f.replace('_qc', '').rsplit('.', 1)[0].lower(): f for f in qc_files}
    auto_map = {f.replace('_auto', '').rsplit('.', 1)[0].lower(): f for f in auto_files}
    
    # Process each pair
    for key, auto_file in auto_map.items():
        qc_file = qc_map.get(key)
        if qc_file:
            auto_path = os.path.join(folder_path, auto_file)
            qc_path = os.path.join(folder_path, qc_file)
            
            try:
                # ✅ Read _qc file (Normal read, first line as header)
                df_qc = pd.read_csv(qc_path)
                
                # ✅ Read _auto file (Keep the first line, use the second line as header for dataframe)
                with open(auto_path, 'r', encoding='utf-8') as f_auto:
                    auto_lines = f_auto.readlines()
                
                df_auto = pd.read_csv(io.StringIO(''.join(auto_lines[1:])))
                
                # ✅ Replace 'Valid QC' column in _auto using _qc's 'Valid QC'
                if 'Valid QC' in df_qc.columns and 'Valid QC' in df_auto.columns:
                    df_auto['Valid QC'] = df_qc['Valid QC'].values
                    
                    # ✅ Combine back with the first line of the original _auto file
                    updated_content = ''.join(auto_lines[:1]) + df_auto.to_csv(index=False, lineterminator='\n')
                    
                    # ✅ Save the modified _auto file (keep the first line and formatting intact, no extra blank lines)
                    with open(auto_path, 'w', encoding='utf-8', newline='') as f_auto_out:
                        f_auto_out.write(updated_content)
                    
                    print(f"Processed: {auto_file}")
                else:
                    print(f"Skipped {auto_file} or {qc_file}: 'Valid QC' column not found.")
            except Exception as e:
                print(f"Error processing {auto_file} and {qc_file}: {e}")
        else:
            print(f"No matching '_qc' file found for: {auto_file}")

# Example: Process files in the current directory
batch_replace_valid_qc('.')
