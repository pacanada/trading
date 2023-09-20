from src.modules.paths import get_project_root
from dataclasses import dataclass
@dataclass
class Config:
    block_size = 50
    n_blocks = 5
    epochs = int(1e4)
    vocab_size = 7 # same as number of classes
    embedding_dim = 5 # must be equal to head_size in this model but not in example
    batch_size=258
    evaluation_steps=100
    n_head=5
    learning_rate=0.001
    dropout=0.1
    load_model = True
    path_model = get_project_root() / "data/weights/transformer_v1.pt"
    num_target = "target_5"
    features = ["open"]
    target = f"label_{num_target}"
    training_ratio = 0.9

    def __post_init__(self):
        if self.embedding_dim%self.n_head!=0:
            raise ValueError(f"Embedding dimension {self.embedding_dim} should be a multiple of n_head={self.n_head}")
config = Config()