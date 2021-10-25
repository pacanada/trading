from src.modules.paths import get_project_root
from src.modules.utils import add_training_type
from src.core.features.utils import feature_pipeline
import time
from datetime import datetime
import pandas as pd
def main(pair_names, input_directory, output_directory, split_size, n_splits):
    for pair_name in pair_names:
        df = pd.read_csv(input_directory / f"{pair_name}.csv")
        df, _ = feature_pipeline(df)
        df = add_training_type(df, split_size, n_splits)
        df.to_csv(output_directory / f"{pair_name}_training.csv")
        print(f"Successfully saved training data for {pair_name}")

if __name__=="__main__":
    pair_names = ["xlmeur", "bcheur","compeur","xdgeur", "etheur", "algoeur", "bateur", "adaeur","xrpeur"]
    input_directory = get_project_root() / "data" / "historical"
    output_directory = get_project_root() / "data" / "training"
    split_size, n_splits = 100,10


    main(pair_names, input_directory, output_directory,split_size, n_splits)