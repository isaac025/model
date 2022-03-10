import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from models.dataset import ProbeDataset
from models.model import NeuralNetwork
import load_dataset.data_proc as dp
from evaluate import train_loop, test_loop

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}  device")

# Dataset & DataLoader
array = dp.load_file_to_nparr('/home/is/Desktop/CURP/model/load_dataset/data/probe_data/probe_pressure_velocity_density_temperature_1_FOM.npy')
(data,targets) = dp.dataset_loader(array)

train_size = int(len(data[0])*.8)
length = len(data[0])

train_data = data[:,0:train_size]
test_data = data[:,(train_size+1):length]

train_targets = targets[0:train_size]
test_targets = targets[train_size:len(targets)]

train = ProbeDataset(train_data,train_targets)
test = ProbeDataset(test_data,test_targets)

train_dataloader = DataLoader(train,batch_size=5)
test_dataloader = DataLoader(test,batch_size=5)

model = NeuralNetwork().to(device)

learning_rate = 1e-3
batch_size = 64
epochs = 10

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n--------------------------------")
    train_loop(train_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model, loss_fn)

print("Done!")
