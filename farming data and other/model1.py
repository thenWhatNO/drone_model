import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")




df = pd.read_csv("farming data and other\data\summary_scores.csv")

fulldata = []
fulldata_y = []

for ind in range(df.shape[0]):
    session = df.iloc[ind]
    session_df = pd.read_csv(session["Session File"])
    game_llist = []
    for i in range(session_df.shape[0]):
        listdd = session_df.iloc[i].tolist()
        game_llist.append(listdd)
    fulldata.append(np.array(game_llist))
    fulldata_y.append(session["Score"])


if not len(fulldata) == len(fulldata_y):
    raise "data x, and data y lens not the same"


def one_hot_encode_action(X, num_classes=6):
    new_out = []
    for y in X:   
        one_hot = np.zeros(num_classes)
        one_hot[int(y)-1] = 1.0
        new_out.append(one_hot)
    return new_out

class MyData():
    def __init__(self, X, Y, class_num):
        self.X = [torch.tensor(x, dtype=torch.float32)for x in X]
        self.Y = []
        for y in np.array(X):
            y = y[:,0]
            y = one_hot_encode_action(y, class_num)
            y.pop(0)
            self.Y.append(y)
        self.Y = [torch.tensor(y, dtype=torch.float32)for y in self.Y]
        print(self.Y[0])
        print("all data converted to tensors")

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, inx):
        X_in = self.X[inx]
        y_in = self.Y[inx]
        return X_in, y_in 

data_set = MyData(fulldata, fulldata_y, 6)
louder = DataLoader(data_set, batch_size=2, shuffle=True)

class dronAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 16)
        self.LSTM = nn.LSTM(16, 64, 2, batch_first=True)
        self.decoder = nn.Sequential(
                nn.Linear(64, 60),
                nn.LeakyReLU(),
                nn.Linear(60, 30),
                nn.LeakyReLU(),
                nn.Linear(30,6)
            )

    def forward(self, x):
        x = self.input(x)
        out, _ = self.LSTM(x)
        decode_in = out[:,-1,:]
        output = self.decoder(decode_in)
        return output
    
model_1 = dronAgent().to(device)


loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)


def train(datalouder, model, loss_fc, optim):
    model.train()
    for batch, (X,y) in enumerate(datalouder):
        X, y = X.to(device), y.to(device)

        print(f"X data {X}, y data {y}")

        optim.zero_grad()

        print(X)
        pred = model(X)
        pred = pred.transpose(1, 2)

        loss = loss_fc(pred, y)
        loss.backward()
        optim.step()

        if batch % 100 ==0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
epoch = 5
for t in range(epoch):
    print(f"epoch{t+1}\n----------------------------")
    train(louder, model_1, loss_f, optimizer)

print("done!")