from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.core.features.utils import add_target
from src.modules.paths import get_project_root
import pandas as pd
from src.transformer_univariate.config import config
from src.transformer_univariate.model import Transformer
from src.transformer_univariate.utils import create_sequences, create_sequences_target, standardize


pairs =["xlmeur", "bcheur","compeur","xdgeur", "etheur", "algoeur", "bateur", "adaeur","xrpeur"]
input_directory = get_project_root() / "data" / "historical_from_20092023"
for pair_name in pairs:
    df = pd.read_csv(input_directory / f"{pair_name}.csv")

    # Assign target
    df = add_target(df, column_to_apply="open", target_list=[20])

    #print(df)
    # dropna
    df = df.dropna()
    bins = [-np.inf]+ config.bins_label+[np.inf]
    labels = [int(i) for i in range(len(bins)-1)]
    
    df[config.target] = pd.cut(df[config.num_target], bins=bins, labels=labels).astype(int)

    # Generate sequence

    X_raw = create_sequences(df[config.features].values, config.block_size)
    X = standardize(X_raw)
    y = create_sequences_target(df[config.target].values, config.block_size)

    # Load model

    m = Transformer(config)
    m.load_state_dict(torch.load(Path(config.path_model) / "weights.pt"))

    # Evaluate model
    m.eval()
    out = m(X.squeeze())
    loss_fn = nn.CrossEntropyLoss()
    loss_all = loss_fn(out.view(-1,config.vocab_size),y.view(-1))
    loss_last = loss_fn(out[:,-1,:], y[:,-1])
    print(f"{pair_name}:Loss all ", loss_all.item(), "Loss last", loss_last.item())

    predicted_class = out[:,-1,:].softmax(-1).argmax(1).detach().numpy()
    target = y[:,-1].detach().numpy()
    print("Accuracy", (predicted_class==target).mean())
    print("Predicted class")
    print(pd.DataFrame(predicted_class).value_counts().to_dict())
    print("Target")
    print(pd.DataFrame(target).value_counts().to_dict())

    # is around 1.3 a good loss for unbalance 6 classes?

    # plt.plot(predicted_class, label="predicted")
    # plt.plot(target, label="target")
    # plt.legend()
    # plt.show()

    # TODO: nan in bateur is because of open price not changing