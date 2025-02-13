import pandas as pd


class DataEntity:
    # DataEntity class is used to store the key, iv_filename, and paired_filename and the dataframes

    # list of fields
    key: str
    iv_filename: str
    paired_filename: str
    output_dir: str
    df_iv: pd.DataFrame
    df_paired: pd.DataFrame

    def __init__(self, key: str, iv_filename: str, paired_filename: str, output_dir: str):
        self.key = key
        self.iv_filename = iv_filename
        self.paired_filename = paired_filename
        self.output_dir = output_dir

    def save_iv(self, name: str):
        """Save the DataFrame at any stage if required."""
        self.df_iv.to_csv(f"{self.output_dir}/{self.iv_filename}_{name}.csv", index=False)

    def save_paired(self, name: str):
        """Save the DataFrame at any stage if required."""
        self.df_paired.to_csv(f"{self.output_dir}/{self.paired_filename}_{name}.csv", index=False)
