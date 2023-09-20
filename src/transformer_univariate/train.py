import torch
import torch.nn as nn

from src.modules.paths import get_project_root
from src.transformer_univariate.config import config
from src.transformer_univariate.model import Transformer

X = torch.load(get_project_root() /"src" / "transformer_univariate" / "data"  / "X_train_all.pt").float()

y = torch.load(get_project_root() /"src" / "transformer_univariate" / "data" / "y_train_all.pt")
X_test = torch.load(get_project_root() /"src" / "transformer_univariate" / "data"  / "X_test_all.pt").float()
y_test = torch.load(get_project_root() /"src" / "transformer_univariate" / "data" / "y_test_all.pt")



print(config)

m = Transformer(config)

#BS = 10
optimizer = torch.optim.AdamW(m.parameters(), lr=0.0005)
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
    if i % 5==0:
        print("i:train", i, "Loss", loss)
        loss_training_l.append(loss.detach().numpy())
        # eval
    if i%5==0:
        with torch.no_grad():
            m=m.eval()
            out = m(X_test.squeeze())
            loss = loss_fn(out.view(-1,config.vocab_size),y_test.view(-1))
            loss_val_l.append(loss.detach().numpy())
            m.train()
            print("i:test", i, "Loss test", loss)

