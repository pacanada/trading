from typing import List
from pathlib import Path

class DataDownloader:
    """Class for downloading from platform of platform client"""
    def __init__(self, pair_names: List[str], directory: Path, platform_client, interval: int):
        self.pair_names = pair_names
        self.directory = directory
        self.platform_client = platform_client
        self.interval = interval

    def download_data(self):
        for pair_name in self.pair_names:
            df = self._init_dataframe(pair_name)
            df_aux = kraken_client.get_last_info_and_preproces(pair_name=pair_name, interval=self.interval)
            df = pd.concat([df, df_aux])
            df = df.drop(["date"], axis=1)
            df = df[~df.index.duplicated(keep='first')].copy()
            df.to_csv(dir_training_data+pair_name+".csv")
    def _init_dataframe(self, pair_name):
        try:
            df = pd.read_csv(dir_training_data+pair_name+".csv", index_col="date")
            df.index = pd.DatetimeIndex(df.index)

        except FileNotFoundError:
            df = pd.DataFrame()
        return df

    