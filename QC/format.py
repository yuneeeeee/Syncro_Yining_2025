import os
import pandas as pd

# Set directory path to current working directory
directory = os.getcwd()

def detect_empty_columns(file_path):
    """Detects the number of leading empty columns in the first row."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    # Count the number of leading tab characters (which indicate empty columns)
    empty_col_count = len(first_line) - len(first_line.lstrip("\t"))
    return empty_col_count

def convert_text_to_columns():
    """Recursively find and process all CSV files in subdirectories."""
    for root, _, files in os.walk(directory):
        csv_files = [f for f in files if f.lower().endswith(".csv")]  # All CSV files
        
        for input_file in csv_files:
            input_path = os.path.join(root, input_file)

            try:
                # Detect number of empty leading columns
                num_empty_cols = detect_empty_columns(input_path)
                
                # Read the file as tab-separated values
                df = pd.read_csv(input_path, delimiter="\t", header=None, dtype=str)  # Read as text to preserve format
                
                # Insert empty columns at the start if detected
                if num_empty_cols > 0:
                    empty_col_df = pd.DataFrame([[""] * num_empty_cols], columns=[f"Unnamed_{i}" for i in range(num_empty_cols)])
                    df = pd.concat([empty_col_df, df], axis=1)  # Insert empty columns

                # Overwrite the original file
                df.to_csv(input_path, index=False, header=False)

                print(f"Converted and replaced: {input_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    convert_text_to_columns()
