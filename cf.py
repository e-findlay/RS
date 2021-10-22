import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import string
import time
import torch
import torch.nn as nn
from nltk.stem.snowball import SnowballStemmer

data = np.load('C:/Users/fox2e/RS/data/cf_data.npy', allow_pickle=True)
data = data.tolist()
data = np.random.permutation(data).tolist()
total = len(data)
train_test_split = round(0.8 * total)
train_data = data[:train_test_split]
test_data = data[train_test_split:]

class EAutoRec(nn.Module):
    def __init__(self, user_dim, ex_dim):
        super(EAutoRec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(user_dim + ex_dim, 50),
        )

        self.decoder = nn.Sequential(
            nn.Linear(50, user_dim)
        )
    def forward(self, x, y):
        h = torch.cat((x,y))
        return self.decoder(torch.sigmoid(self.encoder(h)))

def RMSE_Loss(y_, y, idxs):
    return torch.sqrt(torch.mean((y_ - y) ** 2))

def train(n_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EAutoRec(4055, 4055)
    model = model.to(device)
    criterion = RMSE_Loss
    optimiser = optim.Adam(model.parameters(), lr=0.0001)
    start = time.time()
    loss = 0
    for epoch in range(n_epochs):
        for data in train_data:
            zero_idxs = np.argwhere(np.array(data[0]) == 0)
            rating = torch.FloatTensor(data[0])
            friends = torch.FloatTensor(data[1])
            m1 = np.max(np.array(data[0]))
            m2 = np.max(np.array(data[1]))

            rating = rating.to(device)
            friends = friends.to(device)
            pred = model(rating, friends)

            optimiser.zero_grad()
            loss = criterion(pred, rating, zero_idxs)
            loss.backward()
            optimiser.step()
            
        params = {'model': model.state_dict()}
        torch.save(params, 'C:/Users/fox2e/RS/data/model.pt')
        print(time.time() - start)
        print(loss)
    params = {'model': model.state_dict()}
    torch.save(params, 'C:/Users/fox2e/RS/data/model.pt')
def test():
    model = EAutoRec(4055, 4055)
    params = torch.load('C:/Users/fox2e/RS/data/model.pt')
    model.load_state_dict(params['model'])
    criterion = RMSE_Loss
    LOSS = 0
    for data in test_data:
        zero_idxs = np.argwhere(np.array(data[0]) == 0)
        rating = torch.FloatTensor(data[0])
        explain = torch.FloatTensor(data[1])
        pred = model(rating, explain)
        loss = criterion(pred, rating, zero_idxs)
        LOSS += loss
    print(LOSS/len(test_data))

def calculate_coverage(data):
    pc = 0
    items = [0] * 4055
    model = EAutoRec(4055, 4055)
    params = torch.load('C:/Users/fox2e/RS/data/model.pt')
    model.load_state_dict(params['model'])
    for d in data:
        n_recommended = 0
        n_relevant = 0
        n_recommended_relevant = 0
        zero_idxs = np.argwhere(np.array(d[0]) == 0)
        rating = torch.FloatTensor(d[0])
        explain = torch.FloatTensor(d[1])
        pred = model(rating, explain)
        for i in range(len(pred)):
            if items[i] == 0:
                if pred[i] > 0.6:
                    items[i] = 1
    for i in items:
        if i > 0:
            pc += 1
    print('Prediction Coverage: ', pc / 4055)





def MEP(data):
    model = EAutoRec(4055, 4055)
    params = torch.load('C:/Users/fox2e/RS/data/model.pt')
    model.load_state_dict(params['model'])
    total = 0
    for d in data:
        rating = torch.FloatTensor(d[0])
        explain = torch.FloatTensor(d[1])
        pred = model(rating, explain)
        n_explainable = 0
        n_recommended = 0
        for j in range(len(pred)):
            if pred[j] > 0.6:
                n_recommended += 1
                if explain[j]  > 0:
                    n_explainable += 1
        total += (n_explainable / n_recommended)
    mep = total / len(test_data)
    print(mep)



test()