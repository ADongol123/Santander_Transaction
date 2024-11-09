from pickletools import optimize

import torch
from sklearn import metrics
from torch import device
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions
from dataset import get_data
import torch.utils
import torch.nn.functional as F
from torch.utils.data import DataLoader


class NN(nn.Module):
    def __init__(self,input_size,hidden_dim):
        super(NN,self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.conn1 = nn.Linear(1,hidden_dim)
        self.conn2 = nn.Linear(input_size*hidden_dim,1)
        # self.net = nn.Sequential(
        #     nn.BatchNorm1d(input_size),
        #     nn.Linear(input_size,50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50,1)
        # )

    def forward(self,x):
        print(f"The value of x :{x.shape}")
        BATCH_SIZE = x.shape[0]
        x = self.bn(x)
        # curr x shape: (BATCH_SIZE , 200)
        x = x.view(-1,1) # x is being reshaped to have two dimensions with each element placed on its own row
        # [
        # [1],[2],[3]...[n]
        # ]
        x = F.relu(self.conn1(x)).reshape(BATCH_SIZE,-1) # BATCH_SIZE, input_size * hidden_dim
        return torch.sigmoid(self.conn2(x)).view(-1)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NN(input_size=400,hidden_dim=16).to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=3e-4, weight_decay=1e-5)
loss_fn = nn.BCELoss()

train_ds, val_ds, test_ds, test_ids = get_data()

train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

for epoch in range(15):
    probabilities,  true = get_predictions(val_loader,model,device=DEVICE)
    print(f"Validation ROC: {metrics.roc_auc_score(true,probabilities)}")
    # data,targets = next(iter(train_loader))

    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        scores = model(data)
        # print(scores.shape)
        loss = loss_fn(scores,targets)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


x,y = next(iter(train_loader))
print(x.shape)