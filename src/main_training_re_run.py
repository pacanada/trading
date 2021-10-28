from src.core.training.wandblogger import WandbLogger
from src.core.training.trainer import Trainer
from src.core.training.training_utils import download_training_data, load_model_from_artifact


def load_data(input_folder, input_name):
    df = pd.read_feather(input_folder + input_name)
    df["target_5_multiplied"] = df["target_5"]*1000
    # training, test, validation
    return df[df.type=="training"], df[df.type=="validation"], df[df.type=="validation_unseen"]

if __name__=="__main__":
    file, folder = "training_all_crypto.feather", "temp/"
    project_name = "re_run_divine_snowflake"
    entity_name = "pab_lo4"
    load_model_config = {
        "entity": "pab_lo4",
        "project": "run5_grid_search",
        "model_name": "NeuralNetwork",
        "model_version": "v130",
        "model_format": ".pickle"
        }    
    download_training_data("15VkzDb8sfWTDOl44ODmkNS20KEszWUb-", folder, file)
    df_training, df_test, df_validation = load_data(folder, file)
    columns_features = [col for col in df_training.columns if col.startswith("feature_domain")]
    columns_target = [col for col in df_training.columns if col.startswith("target")]
    columns_target = ["target_5_multiplied"]


    pipe = load_model_from_artifact(**load_model_config)
    print("loaded model: ", pipe)
    try:
        wandb_logger = WandbLogger(project_name=project_name, entity_name=entity_name, model_params=pipe[2].get_params())
    except:   
        wandb_logger = WandbLogger(project_name=project_name, entity_name=entity_name, model_params=pipe[2].get_params())
    trainer = Trainer(pipe, df_training, df_test, df_validation, wandb_logger)

    trainer.init_training(2000, columns_features, columns_target, 5, 20)
    