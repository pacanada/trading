from src.modules.paths import get_project_root
from src.transformer_univariate.model import Transformer
from src.transformer_univariate.config import config
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

m = Transformer(config)
m.load_state_dict(torch.load(Path(config.path_model) / "weights.pt"))

# loade test data
X_test = torch.load(get_project_root() /"src" / "transformer_univariate" / "data"  / "X_test_all.pt")[:30000]
y_test = torch.load(get_project_root() /"src" / "transformer_univariate" / "data" / "y_test_all.pt")[:30000]

print(pd.DataFrame(y_test.view(-1).detach().numpy()).value_counts())

print(X_test.shape)

m.eval()
out = m(X_test.squeeze())
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(out.view(-1,config.vocab_size),y_test.view(-1))
print("Loss test", loss.item())

plt.title("No weight: Density of output")
plt.plot(out.view(-1, config.vocab_size).softmax(-1).mean(0).detach().numpy(), label="output")
plt.plot(pd.DataFrame(y_test.view(-1).detach().numpy()).value_counts(normalize=True).sort_index().values, label="test")
plt.legend()
plt.show()
