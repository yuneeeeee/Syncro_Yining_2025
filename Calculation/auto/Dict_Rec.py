import pandas as pd
import re
import os
import glob

def read_csv_to_dict(filepath):

    df = pd.read_csv(filepath, header=[0, 1])

    # Extract the first and second header rows
    first_header = df.columns.get_level_values(0)
    second_header = df.columns.get_level_values(1).str.strip()

    valid_qc_index = None
    valid_qc_cols = [i for i, name in enumerate(second_header) if re.sub(r'\s+', '', name).lower() == 'validqc']
    if valid_qc_cols:
        valid_qc_index = valid_qc_cols[0]
        df = df[df.iloc[:, valid_qc_index].astype(str).str.strip().str.upper() == 'T']
    else:
        print("Warning: 'Valid QC' column not found. Using all wells.")

    well_id_index = None
    well_id_cols = [i for i, name in enumerate(second_header) if re.sub(r'\s+', '', name).lower() == 'wellid']
    if well_id_cols:
        well_id_index = well_id_cols[0]
        well_ids = df.iloc[:, well_id_index].astype(str).str.strip().str.upper().tolist()
    else:
        print("Warning: 'Well ID' column not found. Using default names.")
        well_ids = [f"Well_{i}" for i in range(len(df))]

    # Get all unique sweep columns from the first header
    sweep_pattern = re.compile(r'Sweep\s*(\d+)', re.IGNORECASE)
    sweeps = list({col.strip() for col in first_header if sweep_pattern.search(col)})
    sweeps = [re.sub(r'\s+', '', sweep) for sweep in sweeps]  # Remove all whitespace from sweep names
    sweeps.sort(key=lambda x: int(sweep_pattern.search(x).group(1)))
    
    data_dict = {}
    for idx, well_id in enumerate(df.index):
        current_well = well_ids[idx]
        data_dict[current_well] = {}

        for sweep_num in sweeps:
            if sweep_num not in data_dict[current_well]:
                data_dict[current_well][sweep_num] = {}

            for col in [c for c in df.columns if sweep_pattern.search(c[0]) and sweep_pattern.search(c[0]).group(1) == sweep_num[-3:]]:
                feature = re.sub(r'[^a-zA-Z0-9]', '', col[1]).lower()
                feature = re.sub(r'(?<![a-zA-Z])\d+$', '', feature)   # Remove trailing numbers
                value = df.iloc[idx][col]
                data_dict[current_well][sweep_num][feature] = value
    
    return data_dict



def get_value(data_dict, well_id, sweep, feature):
    sweep_formatted = re.sub(r'\\s+', '', sweep)
    feature_formatted = re.sub(r'[^a-zA-Z0-9]', '', feature).lower() 
    well_data = data_dict.get(well_id, {})
    value = well_data.get(sweep_formatted, {}).get(feature_formatted, None)
    return value

def get_values_across_sweeps(data_dict, well_id, feature):
    # print(f"Collecting values for Well={well_id}, Feature={feature}")
    feature_formatted = re.sub(r'[^a-zA-Z0-9]', '', feature).lower()
    # print("Formatted feature name:", feature_formatted)
    well_data = data_dict.get(well_id, {})
    # print("Available sweeps in well:", well_data.keys())
    
    values = []
    for sweep_key, sweep_data in well_data.items():
        # print(f"Checking sweep: {sweep_key}, Data: {sweep_data}")
        value = sweep_data.get(feature_formatted)
        if value is not None:
            values.append(value)  # Append each value to the list

    # print("Retrieved values across sweeps:", values)
    return values

def process_all_files():
    all_data = {}
    for filepath in glob.glob("*Recovery*.csv"):
        print(f"Processing file: {filepath}")
        data_dict = read_csv_to_dict(filepath)
        all_data[os.path.basename(filepath)] = data_dict

    return all_data

if __name__ == "__main__":
    all_data = process_all_files()
    if all_data:
        first_file = list(all_data.keys())[0]
        print(get_value(all_data[first_file], 'D02', 'Sweep001', ' Ratio_1'))
        print(get_values_across_sweeps(all_data[first_file], 'D02', ' Ratio_1'))


