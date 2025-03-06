import pandas as pd

def add_valid_qc_column(input_csv, output_csv):
    # Read CSV file
    df = pd.read_csv(input_csv)
    
    # Check if 'Auto QC' column exists
    if 'Auto QC' in df.columns:
        df['Valid QC'] = df['Auto QC'].apply(lambda x: 'T' if x == 1 else 'F')
    else:
        print("Error: 'Auto QC' column not found in the input CSV file.")
        return
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated file saved as {output_csv}")

# Example usage
if __name__ == "__main__":
    input_file = "rundown30%_withpeak_qc.csv"  # Replace with actual input file path
    output_file = "output.csv"  # Replace with desired output file path
    add_valid_qc_column(input_file, output_file)
