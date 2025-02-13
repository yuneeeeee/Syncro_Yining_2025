import os
import re
from enum import Enum
from collections import defaultdict
from typing import Tuple, Dict, Optional, Iterator

class CSV_TYPE(Enum):
    """Enum for different types of CSV files based on filename."""
    IV = "IV"
    Tail = "Tail"
    Recovery = "Recovery"

def detect_csv_type(filename: str) -> CSV_TYPE:
    """Detect the CSV type based on the filename keywords."""
    if "IV" in filename:
        return CSV_TYPE.IV
    elif "Tail" in filename:
        return CSV_TYPE.Tail
    elif "Recovery" in filename:
        return CSV_TYPE.Recovery
    else:
        raise ValueError(f"Unknown CSV type in filename: {filename}")
    
def extract_key(filename: str, regex_expression: str) -> Optional[str]:
    """
    Extracts the unique key from the filename using a regular expression.
    
    :param filename: The CSV file name.
    :param regex_expression: Regular expression to extract the key.
    for example, r"([\d.]+)_(IV|Tail|Recovery)[^\.]*\.csv" will extract the key from filenames like '1.1_IV_2021.csv'
    :return: Extracted key or None if format is incorrect.
    """
    match = re.match(regex_expression, filename)
    return match.group(1) if match else None

def group_paired_files(directory: str, regex_expression: str) -> Iterator[Tuple[str, str, Optional[str]]]:
    """
    Groups CSV files by their common key (e.g., '1.1') and pairs IV with Tail/Recovery.

    :param directory: Folder path containing CSV files.
    :param regex_expression: Regular expression to extract the key from filenames.
    :return: Iterator of tuples (key, IV file path, paired file path)
    """
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    grouped_files: Dict[str, Dict[CSV_TYPE, str]] = defaultdict(dict)
    
    # Categorize files by key
    for file in files:
        key = extract_key(file, regex_expression)
        csv_type = detect_csv_type(file)
        if key and csv_type:
            grouped_files[key][csv_type] = os.path.join(directory, file)
    
    # Yield each group as a tuple (key, IV file, paired file)
    for key, file_dict in grouped_files.items():
        iv_file = file_dict.get(CSV_TYPE.IV)
        paired_file = file_dict.get(CSV_TYPE.Tail) or file_dict.get(CSV_TYPE.Recovery)  # Either Tail or Recovery
        if iv_file and paired_file:  # Only return if both exist
            yield key, iv_file, paired_file

    """ Example usage:

        data/1.1_IV.csv
        data/1.1_Tail.csv
        data/1.2_IV.csv
        data/1.2_Tail.csv
        data/2.1_IV.csv
        data/2.1_Recovery.csv

    Call the function
        group_paired_files("data", r"([\d.]+)_(IV|Tail|Recovery)[^\.]*\.csv")
    Output: 
        Processing Experiment: 1.1
        - IV File: data/1.1_IV.csv
        - Paired File: data/1.1_Tail.csv

        Processing Experiment: 1.2
        - IV File: data/1.2_IV.csv
        - Paired File: data/1.2_Tail.csv
    """
