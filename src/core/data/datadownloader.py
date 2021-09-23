from typing import List
from pathlib import Path
from src.base.platformclient import PlatformClient
import pandas as pd

class DataDownloader:
    """Class for downloading from platform of platform client"""
    def __init__(self, pair_names: List[str], directory: Path, platform_client: PlatformClient, interval: int):
        self.pair_names = pair_names
        self.directory = directory
        self.platform_client = platform_client
        self.interval = interval

    def download_data(self):
        """TODO: check why we need so many steps with dates, i do not remember"""
        for pair_name in self.pair_names:
            df = self._init_dataframe(pair_name=pair_name)
            df_temp = self.platform_client.get_last_historical_data(pair_name=pair_name, interval=self.interval)
            df_temp = self._set_datetime_as_index(df_temp)
            df = pd.concat([df, df_temp])
            df = df.drop(["date"], axis=1)
            df = df[~df.index.duplicated(keep='first')].copy()
            df.to_csv(self.directory / f"{pair_name}.csv")
            print(f"Updated {pair_name=}, total shape={df.shape[0]} ")
        
            

    def _init_dataframe(self, pair_name):
        try:
            df = pd.read_csv(self.directory / f"{pair_name}.csv", index_col="date")
            df.index = pd.DatetimeIndex(df.index)

        except FileNotFoundError:
            df = pd.DataFrame()
        return df


    def _set_datetime_as_index(self, df: pd.DataFrame):
        """TODO: add description"""
        df["date"] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index(pd.DatetimeIndex(df["date"])).copy()
        return df

    