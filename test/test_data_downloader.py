def test_data_downloader_instance():
    from src.core.data.datadownloader import DataDownloader
    from src.modules.paths import get_project_root
    from src.base.platformclient import PlatformClient
    dd = DataDownloader(pair_names=["xlmeur"], directory=get_project_root(), platform_client=PlatformClient )
