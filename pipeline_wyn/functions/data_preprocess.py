import pandas as pd
from functions.data_entity import DataEntity

def basic_preprocess(d: DataEntity, save_df: bool = True) -> DataEntity:
    """Modify CSV format in the data entity's fields and clean up data."""
    # TODO: Implement this function
    df_iv = d.df_iv
    df_paired = d.df_paired

    # Do some pre processing...
    # For example, we might want to drop some columns
    df_iv = df_iv.drop(columns=["Column1", "Column2"])
    df_paired = df_paired.drop(columns=["Column1", "Column2"])

    # Store the preprocessed data in the data entity and return it
    d.df_iv = df_iv
    d.df_paired = df_paired

    if save_df:
        d.save_iv("preprocessed")
        d.save_paired("preprocessed")
    return d

### The below is in case we need to add another preprocess function

# def preprocess2(df: pd.DataFrame) -> pd.DataFrame:
#     """Modify CSV format and clean up data."""
#     # Implement this function
#     return df