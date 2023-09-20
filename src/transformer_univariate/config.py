from pathlib import Path
from src.modules.paths import get_project_root
from dataclasses import dataclass

class Config:
    block_size = 50
    n_blocks = 5
    epochs = int(1e3)
    vocab_size = 7 # same as number of classes
    embedding_dim = 5 # must be equal to head_size in this model but not in example
    batch_size=258
    evaluation_steps=20
    n_head=5
    learning_rate=0.0005
    dropout=0.1
    load_model = True
    run_name = "transformer_v1"
    path_model = str(get_project_root() / f"src/transformer_univariate/models/{run_name}/")
    num_target = "target_5"
    features = ["open"]
    target = f"label_{num_target}"
    training_ratio = 0.9

    def __post_init__(self):
        if self.embedding_dim%self.n_head!=0:
            raise ValueError(f"Embedding dimension {self.embedding_dim} should be a multiple of n_head={self.n_head}")
        # create path if it does not exist
        import os
        if not os.path.exists(Path(Config.path_model)):
            os.makedirs(self.path_model)
        
        
    def dict(self):
        list_of_attributes = [a for a in dir(Config) if not a.startswith('__') and not callable(getattr(Config,a))]
        dict_of_attributes = {k:v for k,v in Config.__dict__.items() if k in list_of_attributes}
        return dict_of_attributes
    
    def json(self):
        import json
        return json.dumps(self.dict())

config = Config()
# print(config)
# # get all attributes to json
# print(config.dict())