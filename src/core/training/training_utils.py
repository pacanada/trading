import wandb
import pickle
import gdown
import os
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