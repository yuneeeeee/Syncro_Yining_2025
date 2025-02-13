from functions.data_entity import DataEntity

def calculate1(d: DataEntity, save_df: bool = True) -> DataEntity:
    """Calculate some results based on the data in the data entity."""
    df_iv = d.df_iv
    df_paired = d.df_paired

    # Calculate some results...
    # For example, we might want to calculate the mean of a column
    mean_iv = df_iv["Column3"].mean()
    mean_paired = df_paired["Column3"].mean()

    # Store the results in the data entity and return it
    d.mean_iv = mean_iv
    d.mean_paired = mean_paired

    if save_df:
        d.save_iv("calculated")
        d.save_paired("calculated")
    return d