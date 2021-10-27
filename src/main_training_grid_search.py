from src.core.training.wandblogger import WandbLogger
from src.core.training.trainer import Trainer
from src.core.training.training_utils import download_training_data, load_model_from_artifact
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import numpy as np

def load_data(input_folder, input_name):
    df = pd.read_feather(input_folder + input_name)
    df["target_5_multiplied"] = df["target_5"]*1000
    # training, test, validation
    return df[df.type=="training"], df[df.type=="validation"], df[df.type=="validation_unseen"]

if __name__=="__main__":
    file, folder = "training_all_crypto.feather", "temp/"
    project_name = "run5_grid_search"
    entity_name = "pab_lo4"
   
    #download_training_data("15VkzDb8sfWTDOl44ODmkNS20KEszWUb-", folder, file)
    df_training, df_test, df_validation = load_data(folder, file)
    df_training = df_training.sample(10000, random_state=1).copy()
    columns_features = [col for col in df_training.columns if col.startswith("feature_domain")]
    columns_target = [col for col in df_training.columns if col.startswith("target")]
    columns_target = ["target_5_multiplied"]

    hidden_layer_sizes_list = [(15,15), (50,50)]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    alphas = [0.001,0.0001]

    list_params = [(hidden_layer_sizes, learning_rate, alpha) for hidden_layer_sizes in hidden_layer_sizes_list for learning_rate in learning_rates for alpha in alphas]




    for index, (hidden_layer_sizes, learning_rate, alpha) in enumerate(list_params):
        print(f"{index} / {len(list_params)}--Training for ",hidden_layer_sizes, learning_rate, alpha )
        nn = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate,
            warm_start=True,
            random_state=1,
            max_iter=5,
            verbose=False,
            tol=1e-8,
            alpha=alpha,
            )
        pipe = Pipeline([("std_scaler", StandardScaler()),
                        ("to_float32",FunctionTransformer(np.float32)),
                        ("nn", nn)]
                    )
        try:
            wandb_logger = WandbLogger(project_name=project_name, entity_name=entity_name, model_params=pipe[2].get_params())
        except:   
            wandb_logger = WandbLogger(project_name=project_name, entity_name=entity_name, model_params=pipe[2].get_params())
        trainer = Trainer(pipe, df_training, df_test, df_validation, wandb_logger)

        trainer.init_training(100, columns_features, columns_target, 2, 20)