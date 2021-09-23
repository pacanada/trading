from src.core.data.datadownloader import DataDownloader
from src.modules.paths import get_project_root
from src.kraken.krakenclient import KrakenClient
import time
from datetime import datetime

def main(pair_names, directory, platform_client, interval, sleep_time):
    try:
        while True:
            dd = DataDownloader(pair_names, directory, platform_client)
            dd.download_data(interval=interval)
            print(f"Updated in timestamp {datetime.now()}")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Keyboard interrupt")

if __name__=="__main__":
    pair_names = ["xlmeur", "bcheur","compeur","xdgeur", "etheur", "algoeur", "bateur", "adaeur","xrpeur"]
    directory = get_project_root() / "data" / "historical"
    platform_client = KrakenClient()
    interval = 1
    sleep_time = 1*60*60

    main(pair_names, directory, platform_client, interval, sleep_time)