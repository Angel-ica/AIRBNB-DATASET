import torch
import numpy as np
import pandas as pd
import yaml 
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# PyTorch dataset that returns a tuple of features, label when indexed
class AirbnbNightlyPriceDataset(Dataset):
    def __init__(self):
        super(AirbnbNightlyPriceDataset, self).__init__()
        df = pd.read_csv('/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/clean_data.csv')
        self.X=df.drop('Price_Night',axis=1)
        self.X, self.y = df.select_dtypes(include=['float', 'int']), df['Price_Night']
        self.n_samples = df.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values.astype(np.float32)
        X_tensor = torch.tensor(X)
        y = self.y[idx]
        y_tensor = torch.tensor(y)
        return X_tensor, y_tensor

dataset = AirbnbNightlyPriceDataset()
# sample = dataset[10]
# features, label = sample
# print(features, label)

def data_split(dataset):
    train_set, test_set = random_split(dataset, [0.7,0.3])
    train_set, val_set = random_split(train_set, [0.5,0.5])
    return train_set, test_set, val_set

def data_loader(dataset, batch_size):
    train, test, val = data_split(dataset)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_sampler=batch_size, shuffle=True)
    return train_dl, test_dl, val_dl

DL=data_loader(dataset,64)


class TrainNN(torch.nn.Module):
    def __init__(self, config):
        super(TrainNN,self).__init__() 
        config = get_nn_config()
        hidden_layer_width = config[-2]
        depth = config[-1]
        self.linear_layer = []
        input_nodes = 11
        output_node = 11
        for hidden_layer in range(depth -1 ):
            self.linear_layer.append(torch.nn.Linear(input_nodes, hidden_layer_width))
            self.linear_layer.append(torch.nn.ReLU())
            input_nodes = hidden_layer_width
        self.linear_layer.append(torch.nn.Linear(input_nodes, output_node))
        self.linear_layer = torch.nn.Sequential(*self.linear_layer)
    
    def forward(self, X):
        return self.linear_layer(X)
model = TrainNN(config)

def train(model, dataloader, epochs=2):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    batch_idx = 0
    for epoch in range(epochs):
      for batch in dataloader[0]:
        X, y = batch
        y = torch.unsqueeze(y, 1)
        prediction = model(X)
        loss = F.mse_loss(prediction, y.float())
        print(loss.item())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        writer.add_scalar('loss', loss.item(), batch_idx)
        batch_idx += 1

      with torch.no_grad():
          avg_val_loss = 0
          for batch in dataloader[-1]:
              X, y = batch
              y = torch.unsqueeze(y, 1)
              prediction = model(X)
              val_loss = F.mse_loss(prediction, y.float())
              avg_val_loss += val_loss.item()

          avg_val_loss /= len(dataloader[-1])
          writer.add_scalar('val_loss', avg_val_loss, epoch)

train(model, DL)

def get_nn_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config



def save_model():

    pass



