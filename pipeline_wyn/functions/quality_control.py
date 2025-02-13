import pandas as pd
from functions.data_entity import DataEntity

def qc1(d: DataEntity, save_df: bool = True) -> DataEntity:
    """Perform quality control check 1, for iv data."""
    df_iv = d.df_iv

    # implement the quality control check
    # enter code here blabla
    
    # Store the results in the data entity and return it
    d.df_iv = True  # Placeholder value

    if save_df:
        d.save_iv("qc1") # only save iv data
    return d


def qc2(d: DataEntity, save_df: bool = True) -> DataEntity:
    """Perform quality control check 2, for paired data."""
    df_paired = d.df_paired

    # implement the quality control check
    # enter code here blabla
    
    # Store the results in the data entity and return it
    d.df_paired = True  # Placeholder value

    if save_df:
        d.save_paired("qc2") # only save paired data
    return d