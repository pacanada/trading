import pandas as pd
from src.modules.paths import get_project_root
import numpy as np
import torch
from src.transformer_univariate.config import config
from src.transformer_univariate.utils import create_sequences, create_sequences_target, standardize

def main():
    df = pd.read_feather(get_project_root() / "data" / "training" / "training_all_crypto_14_12_2021.feather")

    # Add classification target
    bins_no_inf = df[config.num_target].quantile([0.025, 0.15, 0.5 , 0.85, 0.975]).to_list()
    # [-0.006509775145447816, -0.002259122887203158, 0.0, 0.0023185157608664464, 0.006562980687062181]
    # print("min", df[config.num_target].min())
    # print("max", df[config.num_target].max())
    # print(bins_no_inf)
    bins = [-np.inf]+bins_no_inf+[np.inf]
    labels = [int(i) for i in range(len(bins)-1)]

    df[config.target] = pd.cut(df[config.num_target], bins=bins, labels=labels).astype(int)

    # Generate sequences normalized by the mean and std of the set defined by the pair_name and the sequence length (block_size)
    X_train_all = torch.empty((0, config.block_size, len(config.features)), dtype=torch.float16)
    X_test_all = torch.empty((0, config.block_size, len(config.features)), dtype=torch.float16)

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

    # Serialize all variables to pytorch
    torch.save(X_train_all, get_project_root() /"src" / "transformer_univariate" / "data"  / "X_train_all.pt")
    torch.save(y_train_all, get_project_root() /"src" / "transformer_univariate" / "data" / "y_train_all.pt")
    torch.save(X_test_all, get_project_root() /"src" / "transformer_univariate" / "data"  / "X_test_all.pt")
    torch.save(y_test_all, get_project_root() /"src" / "transformer_univariate" / "data" / "y_test_all.pt")




if __name__ == "__main__":
    main()