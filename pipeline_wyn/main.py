import os

from functions.read_folder import group_paired_files
from functions.data_checkin import basic_checkin
from functions.data_preprocess import basic_preprocess
from functions.quality_control import qc1, qc2
from functions.calculations import calculate1
from functions.pipeline import Pipeline
from functions.data_entity import DataEntity

# Define the steps of the pipeline
pipeline = Pipeline(
    steps=[
        basic_checkin,
        basic_preprocess,
        qc1, 
        qc2,
        calculate1
        # or, if you don't want to save the results of any step, 
        # please set save_df=False like uncommenting the line below
        # lambda d: qc1(d, save_df=False), 
    ]
)

# Example: Group files in the "data/" folder
folder_path = "data/"
regex_expression = r"([\d.]+)_(IV|Tail|Recovery)[^\.]*\.csv" # Extracts the key from filenames like '1.1_IV_2021.csv'
output_dir = "output/"

# check if the output directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# clear the output directory before running the pipeline, comment the following lines if you want to keep the files
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))

# Load and preprocess the data
data_entities = []
for key, iv_file, paired_file in group_paired_files(folder_path, regex_expression):
    print(f"\nProcessing Experiment: {key}")
    print(f"  - IV File: {iv_file}")
    print(f"  - Paired File: {paired_file}")
    data_entity = DataEntity(key, iv_file, paired_file, output_dir)

    # Run the pipeline on each data entity
    processed_data_entity = pipeline.run(data_entity)

    data_entities.append(processed_data_entity)

# Plot the results
# sort the data_entities by key before plotting by i.e.,
for data_entity in sorted(data_entities, key=lambda x: x.key):
    pass
    ### To be finished