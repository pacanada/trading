from typing import List
from pathlib import Path

class DataDownloader:
    """Class for downloading from platform of platform client"""
    def __init__(self, pair_names: List[str], directory: Path, platform_client):
        self.pair_names = pair_names
        self.directory = directory
        self.platform_client = platform_client

    def download_data(self):
        pass
    def _init_dataframe(self, pair_name):
        try:
            df = pd.read_csv(dir_training_data+pair_name+".csv", index_col="date")
            df.index = pd.DatetimeIndex(df.index)

        except FileNotFoundError:
            df = pd.DataFrame()
        return df

    