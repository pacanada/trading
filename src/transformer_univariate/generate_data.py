import pandas as pd
from src.core.features.utils import add_target
from src.modules.paths import get_project_root
import numpy as np
import torch
from src.transformer_univariate.config import config
from src.transformer_univariate.utils import create_sequences, create_sequences_target, standardize
import os

KAGGLE_DATASET = True

def main():
    if KAGGLE_DATASET:
        df = pd.DataFrame()
        for file in list(os.walk(get_project_root() / "data" / "kaggle_dataset/1min/"))[0][2]:
            df_aux = pd.read_parquet(get_project_root() / "data" / "kaggle_dataset/1min" / file).reset_index()
            df_aux["pair_name"] = file.split(".")[0].lower()
            df_aux = add_target(df_aux, column_to_apply="open", target_list=[20])
            df = pd.concat([df, df_aux[["open_time", "open", "pair_name", "target_20"]]])
        df = df.rename(columns={"open_time": "date"})
        df = df.dropna()

        df.to_parquet(get_project_root() / "data" / "kaggle_dataset" / "kaggle_dataset.parquet")
           
            

    else:
        df = pd.read_feather(get_project_root() / "data" / "training" / "training_all_crypto_14_12_2021.feather")

    # Add classification target
    bins_no_inf = df[config.num_target].quantile([0.025, 0.15, 0.5 , 0.85, 0.975]).to_list()
    print(bins_no_inf)
    # with [0.025, 0.15, 0.5 , 0.85, 0.975]
    # [-0.012927892170363052, -0.0048195501696362275, 0.0, 0.004946431814054853, 0.013178091286970997]
    # print("min", df[config.num_target].min())
    # print("max", df[config.num_target].max())
    # print(bins_no_inf)
    bins = [-np.inf]+bins_no_inf+[np.inf]
    labels = [int(i) for i in range(len(bins)-1)]

    df[config.target] = pd.cut(df[config.num_target], bins=bins, labels=labels).astype(int)

    # Generate sequences normalized by the mean and std of the set defined by the pair_name and the sequence length (block_size)
    X_train_all = torch.empty((0, config.block_size, len(config.features)), dtype=torch.float32)
    X_test_all = torch.empty((0, config.block_size, len(config.features)), dtype=torch.float32)

    y_train_all = torch.empty((0, config.block_size), dtype=torch.uint8)
    y_test_all = torch.empty((0, config.block_size), dtype=torch.uint8)

    for name, group in df.groupby("pair_name"):
        print("Generating trainining data for", name)
        X_raw = create_sequences(group[config.features].values, config.block_size)
        y = create_sequences_target(group[config.target].values, config.block_size)

        X = standardize(X_raw)


        length = len(X)
        idx_train = int(length*config.training_ratio)   
        X_train, y_train = X[:idx_train], y[:idx_train]
        X_test, y_test = X[idx_train:], y[idx_train:]
        X_train_all = torch.cat((X_train_all, X_train), dim=0)
        y_train_all = torch.cat((y_train_all, y_train), dim=0)
        X_test_all = torch.cat((X_test_all, X_test), dim=0)
        y_test_all = torch.cat((y_test_all, y_test), dim=0)
    if KAGGLE_DATASET:
        directory = get_project_root() / "src" / "transformer_univariate" / "data" / "kaggle"
    else:
        directory = get_project_root() / "src" / "transformer_univariate" / "data" 

    # Serialize all variables to pytorch
    torch.save(X_train_all, directory / "X_train_all.pt")
    torch.save(y_train_all, directory / "y_train_all.pt")
    torch.save(X_test_all, directory / "X_test_all.pt")
    torch.save(y_test_all, directory / "y_test_all.pt")




if __name__ == "__main__":
    main()