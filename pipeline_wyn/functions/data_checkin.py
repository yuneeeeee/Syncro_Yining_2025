import pandas as pd
from functions.data_entity import DataEntity


def load_csv(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path, header=[0, 1])

def validate_csv_structure(df: pd.DataFrame) -> bool:
    """Check if CSV has the expected header structure."""
    if df.shape[0] < 3:  # At least 3 rows (2 headers + data)
        return False
    required_columns = ["Well ID", "Valid QC"] # the real logic might be more complex but this is just a placeholder
    return all(col in df.iloc[1].values for col in required_columns)

def check_filename_rules(filename: str) -> bool:
    """Ensure filename contains expected keywords (IV, Tail, Recovery)."""
    return any(keyword in filename for keyword in ["IV", "Tail", "Recovery"])


def basic_checkin(d: DataEntity, save_df: bool = False) -> DataEntity:
    """
    Load and validate the data, and store it in the data entity.
    Optionally save the dataframes to CSV files.
    """
    # Load the data
    df_iv = load_csv(d.iv_filename)
    df_paired = load_csv(d.paired_filename)
    
    # Validate the data
    if not validate_csv_structure(df_iv) or not validate_csv_structure(df_paired):
        raise ValueError("Invalid CSV structure detected.")
    
    # Check the filenames
    if not check_filename_rules(d.iv_filename) or not check_filename_rules(data_entity.paired_filename):
        raise ValueError("Invalid filename detected.")
    
    # Store the data in the data entity
    d.df_iv = df_iv
    d.df_paired = df_paired

    if save_df:
        d.save_iv("checked")
        d.save_paired("checked")
    
    return d