import gdown
import os
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from typing import Union
import copy
import wandb
import pickle
import os
import json

class WandbLogger:
    def __init__(self, project_name:str, entity_name:str, model_params:dict):
        self.run = wandb.init(config=model_params,reinit=True, project=project_name, entity=entity_name)
    def log(self, kwargs):
        self.run.log(kwargs)
    def log_artifact(self, name:str, model, metrics_dict):
        temp_dir = Path("temp/")
        os.makedirs(temp_dir, exist_ok=True)
        
        pickle.dump(model, open(temp_dir / f"{name}.pickle", 'wb'))
        json.dump(metrics_dict, open(temp_dir /f"{name}.json", 'w'))
        self.artifact = wandb.Artifact(name=name, type='model')
        self.artifact.add_file(temp_dir /f"{name}.pickle")
        self.artifact.add_file(temp_dir /f"{name}.json")
        self.run.log_artifact(self.artifact)


class Trainer:
    def __init__(self,
                 model,
                 training_data:pd.DataFrame,
                 test_data:pd.DataFrame,
                 validation_data:pd.DataFrame,
                wandb_logger: WandbLogger):
        self.model = copy.deepcopy(model)
        self.training_data= training_data
        self.test_data=test_data
        self.validation_data=validation_data
        self.score_training = []
        self.score_test = []
        self.score_validation = []
        self.wandb_logger = wandb_logger
    def get_training_info_dict(self, it):
        training_info_dict = {"epochs": it,
                     "score_training":self.score_training[-1:][0],
                    "score_test": self.score_test[-1:][0],
                    "score_validation": self.score_validation[-1:][0]}
        return training_info_dict
    def init_training(self, epochs, features, target, save_every=10, print_every=10):
        threshold_val =-0.1
        
        for it in range(epochs):
            self.model.fit(self.training_data[features], self.training_data[target].values.ravel())
            if it%save_every==0:
                self.score_training.append(self.model.score(self.training_data[features],self.training_data[target]))
                self.score_test.append(self.model.score(self.test_data[features],self.test_data[target]))
                self.score_validation.append(self.model.score(self.validation_data[features],self.validation_data[target]))
            if it%print_every==0:
                training_info_dict = self.get_training_info_dict(it=it)
                print(training_info_dict)
                self.wandb_logger.log(training_info_dict)

            if (self.score_validation[-1:][0] > threshold_val) | (epochs==(it+1)):
                training_info_dict = self.get_training_info_dict(it=it)
                self.wandb_logger.log(training_info_dict)
                print("Saved model")
                self.wandb_logger.log_artifact(model=copy.deepcopy(self.model), name="NeuralNetwork", metrics_dict=training_info_dict)
                self.best_model = copy.deepcopy(self.model)
                threshold_val = self.score_validation[-1:][0]
def load_model_from_artifact(entity, project, model_name, model_version, model_format):
    run = wandb.init(entity=entity, project=project)
    artifact = run.use_artifact(f"{entity}/{project}/{model_name}:{model_version}", type='model')
    artifact_dir = artifact.download()
    file = open(artifact_dir + f"/{model_name}{model_format}" ,'rb')
    model = pickle.load(file)
    return model
        
def download_training_data(id_file, output_folder, output_name):
    url = f"https://drive.google.com/uc?id={id_file}"
    os.makedirs(output_folder, exist_ok=True)
    gdown.download(url, output_folder + output_name, quiet=False)
    
def score_predictions(model, df_training, df_validation, df_validation_unseen, columns_features, columns_targets):
    training_score = model.score(df_training[columns_features], df_training[columns_target])
    validation_score = model.score(df_validation[columns_features], df_validation[columns_target])
    validation_unseen_score = model.score(df_validation_unseen[columns_features], df_validation_unseen[columns_target])
    return training_score, validation_score, validation_unseen_score
def load_data(input_folder, input_name):
    df = pd.read_feather(input_folder + input_name)
    df["target_5_multiplied"] = df["target_5"]*1000
    # training, test, validation
    return df[df.type=="training"], df[df.type=="validation"], df[df.type=="validation_unseen"]
if __name__=="__main__":
    file, folder = "training_all_crypto.feather", "temp/"
    project_name = "run5"
    entity_name = "pab_lo4"
    load_previous_model = False
    load_model_config = {"entity": "pab_lo4", "project": "run4", "model_name": "NeuralNetwork", "model_version": "v18", "model_format": ".pickle"}    
    download_training_data("15VkzDb8sfWTDOl44ODmkNS20KEszWUb-", folder, file)
    df_training, df_test, df_validation = load_data(folder, file)
    columns_features = [col for col in df_training.columns if col.startswith("feature_domain")]
    columns_target = [col for col in df_training.columns if col.startswith("target")]
    columns_target = ["target_5_multiplied"]

    hidden_layer_sizes_list = [(10,10), (15,15), (20,20), (10,10,10)]

    if load_previous_model:
        pipe = load_model_from_artifact(**load_model_config)
        print("loaded model: ", pipe)
    else:
        for hidden_layer_sizes in hidden_layer_sizes_list:
            nn = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=0.001,
                warm_start=True,
                random_state=1,
                max_iter=5,
                verbose=False,
                tol=1e-8)
            pipe = Pipeline([("std_scaler", StandardScaler()),
                            ("to_float16",FunctionTransformer(np.float16)),
                            ("nn", nn)]
                        )
    try:
        wandb_logger = WandbLogger(project_name=project_name, entity_name=entity_name, model_params=pipe[2].get_params())
    except:   
        wandb_logger = WandbLogger(project_name=project_name, entity_name=entity_name, model_params=pipe[2].get_params())
    trainer = Trainer(pipe, df_training, df_test, df_validation, wandb_logger)

    trainer.init_training(500, columns_features, columns_target, 5, 20)