import torch
import numpy as np
import pandas as pd
import yaml 
import json
import os
import math
import time 
import itertools
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tabular_data import load_airbnb
from time import strftime

# PyTorch dataset that returns a tuple of features, label when indexed
class AirbnbNightlyPriceDataset(Dataset):
    def __init__(self):
        super(AirbnbNightlyPriceDataset, self).__init__()
        df = pd.read_csv('/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/airbnb-property-listings/tabular_data/clean_data.csv')
        df.drop('Unnamed: 19', axis=1,inplace=True)
        print(df['bedrooms'])
        category = df['Category']
        encoded = pd.get_dummies(category)
        self.X, self.y = load_airbnb(df,"bedrooms")
        self.X = df.select_dtypes(include=['float', 'int'])
        self.X = pd.concat([self.X,encoded],axis =1)
        self.y = pd.to_numeric(self.y, errors='coerce')
        self.n_samples = df.shape[0]
        print(self.X, self.y)
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values.astype(np.float32)
        X_tensor = torch.tensor(X)
        y = self.y[idx].astype(np.int64)
        y_tensor = torch.tensor(y)
        return X_tensor, y_tensor

dataset = AirbnbNightlyPriceDataset()
sample = dataset[10]
features, label = sample
print(features.shape)

def data_split(dataset):
    train_set, test_set = random_split(dataset, [0.7,0.3])
    train_set, val_set = random_split(train_set, [0.5,0.5])
    return train_set, test_set, val_set

def data_loader(dataset, batch_size):
    train, test, val = data_split(dataset)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)
    return train_dl, test_dl, val_dl

# DL=data_loader(dataset,32)


def get_nn_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config

class TrainNN(torch.nn.Module):
    def __init__(self, config):
        super(TrainNN,self).__init__() 
        config = get_nn_config('/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/Pytorch/nn_config.yaml')
        hidden_layer_width = config['hidden-layer-width']
        depth = config['depth']
        self.linear_layer = []
        input_nodes = 16
        output_node = 1
        self.linear_layer.append(torch.nn.Linear(input_nodes, hidden_layer_width))
        for hidden_layer in range(depth -1 ):
            self.linear_layer.append(torch.nn.Linear(hidden_layer_width, hidden_layer_width))
            self.linear_layer.append(torch.nn.ReLU())
        self.linear_layer.append(torch.nn.Linear(hidden_layer_width, output_node))
        self.linear_layer = torch.nn.Sequential(*self.linear_layer)
    
    def forward(self, X):
        return self.linear_layer(X)

model = TrainNN('/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/Pytorch/nn_config.yaml')

def save_model(model, optimiser, optimiser_parameters, performance_metrics ):
    if not isinstance(model, torch.nn.Module):
        print("Not a Pytorch module")
    nn = '/Users/angelicaaluo/Airbnb/AIRBNB-DATASET/Pytorch/NN_bedrooms/'
    if not os.path.isdir(nn):
        os.mkdir(nn)
    nn_Regression = f'{nn}Regression'
    if not os.path.isdir(nn_Regression):
        os.mkdir(nn_Regression)
    time = strftime('%Y-%m-%d_%H:%M:%S')
    optimiser_class = optimiser.__class__.__name__
    model_folder = f'{nn_Regression}/{optimiser_class}'
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    timed_model_folder = f'{model_folder}/{time}'
    if not os.path.isdir(timed_model_folder):
        os.mkdir(timed_model_folder)
    model_path = f'{timed_model_folder}/model.pt'
    sd = model.state_dict()
    torch.save(sd,model_path)

    optimiser_path = f'{timed_model_folder}/hyperparameters.json'
    with open (optimiser_path,'w') as f:
        json.dump(optimiser_parameters, f)
    
    metrics_path = f'{timed_model_folder}/metrics.json'
    if not os.path.isfile(metrics_path,):
        with open (metrics_path,'w') as f:
            json.dump(performance_metrics, f)

    print('done ')

def train(model,optimiser_class, learning_rate, dataloader, epochs):
    optimizer_ins = getattr(torch.optim, optimiser_class)
    optimiser = optimizer_ins(model.parameters(),learning_rate)
    writer = SummaryWriter()
    batch_idx = 0
    start_time = time.time()
    for epoch in range(epochs):
      for batch in dataloader[0]:
        X, y = batch
        y = torch.unsqueeze(y, 1)
        pred_start_time = time.time()
        prediction = (model(X))
        pred_end_time = time.time()
        loss = (F.mse_loss(prediction, y.float()))
        RMSE_loss = math.sqrt(loss)
        print(loss.item())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        writer.add_scalar('loss', loss.item(), batch_idx)
        batch_idx += 1
    end_time = time.time()

    with torch.no_grad():
        avg_val_loss = 0
        for batch in dataloader[-1]:
            X, y = batch
            y = torch.unsqueeze(y, 1)
            prediction = model(X)
            val_loss = F.mse_loss(prediction, y.float())
            avg_val_loss += val_loss.item()
            val_RMSE = math.sqrt(loss)

        avg_val_loss /= len(dataloader[-1])
        writer.add_scalar('val_loss', avg_val_loss, epoch)

    training_duration = (end_time - start_time)
    inference_latency = (pred_start_time - pred_end_time)
    eval_metrics = {
        'train_RMSE': RMSE_loss,
        'val_RMSE : ': val_RMSE,
        'training_duration' : training_duration,
        'inference_latency' : inference_latency
    }
    optimiser_parameters = {
        'optimiser_class' : optimiser_class,
        'learning_rate' : learning_rate
    }

    return model, optimiser,optimiser_parameters, eval_metrics


def generate_nn_configs():
    hyperparameters_list=[]
    different_values = {
        'optimiser_class' : ['Adadelta','Adam','SGD'],
        'learning_rate' : [0.001, 0.1, 0.03],
        'hidden_layer_width' : [4,5,5],
        'depth' : [4,5,5]
    }
    keys, values = different_values.keys(), different_values.values()
    for p_vals in itertools.product(*values):
        hyperparameters = dict(zip(keys,p_vals))
        hyperparameters_list.append(hyperparameters)
    return hyperparameters_list
    
def find_best_nn():
    dataloader = data_loader(dataset, 35)
    hyperparameters = generate_nn_configs()
    lowest_val_rmse = np.inf
    best_model = None
    for single_dict in hyperparameters:
        optimiser_class = single_dict['optimiser_class']
        learning_rate = single_dict['learning_rate']
        trained_model = train(model, optimiser_class, learning_rate, dataloader, 16)
        performance_metrics = trained_model[-1]
        val_rmse = performance_metrics['val_RMSE : ']
        if val_rmse < lowest_val_rmse:
            lowest_val_rmse = val_rmse
            best_model = trained_model
    return best_model

def save_best_model():
    best_model = find_best_nn()
    model, optimizer_class, hyperparameters, metrics = best_model
    save_model(model, optimizer_class, hyperparameters, metrics)


if __name__ == '__main__':
    find_best_nn()
    save_best_model()