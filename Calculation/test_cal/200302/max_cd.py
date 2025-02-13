import pandas as pd
import numpy as np
import os

input_files = ['-75IV_Plate1.csv', '-75IV_Plate2.csv', '-75IV_Plate3.csv', '-100IV_Plate1.csv', '-100IV_Plate2.csv', '-100IV_Plate3.csv']
output_files = ['max_cd_75_Plate1.csv', 'max_cd_75_Plate2.csv','max_cd_75_Plate3.csv', 'max_cd_100_Plate1.csv', 'max_cd_100_Plate2.csv', 'max_cd_100_Plate3.csv']

def process_file(input_path, output_path):
    df = pd.read_csv(input_path, header=[0, 1])
    
    df.columns = pd.MultiIndex.from_tuples([(col1.strip(), col2.strip()) for col1, col2 in df.columns])

    tp1_curden_columns = [(f'Sweep {i:03d}', 'TP1CurDen') for i in range(1, 18)]
    tp2_curden_columns = [(f'Sweep {i:03d}', 'TP2CurDen') for i in range(1, 18)]
    Parameter_column = ('Sweep Results', 'Parameter')

    all_columns = [Parameter_column] + tp1_curden_columns + tp2_curden_columns
    
    missing_columns = [col for col in all_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns in file {input_path}: {missing_columns}")
        return

    df_extracted = df.loc[:, all_columns]

    df_extracted.columns = ['Parameter'] + [f'TP1CurDen_{i}' for i in range(1, 18)] + [f'TP2CurDen_{i}' for i in range(1, 18)]

    def get_largest_absolute(df, column_prefix, num_columns):
        columns = [f'{column_prefix}_{i}' for i in range(1, num_columns + 1)]
        return df[columns].apply(lambda row: row.loc[row.abs().idxmax()] if not row.isna().all() else np.nan, axis=1)

    def get_compound_name(df):
        df_extracted = df.loc[:, [('Sweep Results', 'Parameter'), ('Sweep 001', 'Compound Name')]]
        df_extracted.columns = ['Parameter', 'Compound Name']
        return df_extracted

    df_extracted['TP1_Max_CurDen'] = get_largest_absolute(df_extracted, 'TP1CurDen', 17)
    df_extracted['TP2_Max_CurDen'] = get_largest_absolute(df_extracted, 'TP2CurDen', 17)

    df_result = df_extracted[['Parameter', 'TP1_Max_CurDen', 'TP2_Max_CurDen']]

    compound_name_df = get_compound_name(df)
    df_result = pd.merge(df_result, compound_name_df, on='Parameter', how='left')

    df_result.to_csv(output_path, index=False)
    print(f"Processed {input_path} and saved the result to {output_path}")

for input_file, output_file in zip(input_files, output_files):
    process_file(input_file, output_file)

print("All files processed.")