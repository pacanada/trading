from pathlib import Path
import torch
import torch.nn as nn

from src.modules.paths import get_project_root
from src.transformer_univariate.config import config
from src.transformer_univariate.model import Transformer

X = torch.load(get_project_root() /"src" / "transformer_univariate" / "data"  / "X_train_all.pt")

y = torch.load(get_project_root() /"src" / "transformer_univariate" / "data" / "y_train_all.pt")
X_test = torch.load(get_project_root() /"src" / "transformer_univariate" / "data"  / "X_test_all.pt")[:1000]
y_test = torch.load(get_project_root() /"src" / "transformer_univariate" / "data" / "y_test_all.pt")[:1000]


print(config.dict())

m = Transformer(config)
if config.load_model:
    m.load_state_dict(torch.load(Path(config.path_model) / "weights.pt"))
    print("Model loaded from", Path(config.path_model) / "weights.pt")

#BS = 10
optimizer = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

loss_training_l = []
loss_val_l = []

for i in range(config.epochs):
    idx=torch.randint(0, X.size()[0], (config.batch_size,))
    #m.zero_grad()
    optimizer.zero_grad()
    out = m(X[idx].squeeze())
    # some broadcasting magic (from mingpt)
    loss = loss_fn(out.view(config.batch_size*config.block_size,config.vocab_size),y[idx].view(config.batch_size*config.block_size))
    loss.backward()
    optimizer.step()
    if i % config.evaluation_steps==0:
        loss_training = loss.item()
        loss_training_l.append(loss_training)
        with torch.no_grad():
            m=m.eval()
            out = m(X_test.squeeze())
            loss = loss_fn(out.view(-1,config.vocab_size),y_test.view(-1))
            loss_val = loss.item()
            loss_val_l.append(loss_val)
            m.train()
            print("i:", i, "Loss train", loss_training, "Loss val", loss_val)
            torch.save(m.state_dict(), Path(config.path_model) / "weights.pt")

# save model
torch.save(m.state_dict(), Path(config.path_model) / "weights.pt")
# save config as json
import json
with open(Path(config.path_model)  / "config.json", "w") as f:
    json.dump(config.json(), f)

# save losses
import matplotlib.pyplot as plt
plt.plot(loss_training_l, label="training")
plt.plot(loss_val_l, label="validation")
plt.legend()
plt.savefig(Path(config.path_model) / "loss.png")