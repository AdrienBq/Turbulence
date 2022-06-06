import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import netCDF4 as nc
import shap

import os
import sys
from pathlib import Path

os.chdir(Path(sys.path[0]).parent)
import modules.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F


#----------------PREPARE DATA----------------

coarse_factors = [32]
Directory = f"data"

variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
nz=376

len_in = nz*(len(variables)-1)
len_out = nz

model_number = 11
tmin=1
tmax=62+1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('using cuda : ', torch.cuda.is_available())

path_times_train = f'data/test_train_times/times_train_{model_number}.csv'
path_times_test = f'data/test_train_times/times_test_{model_number}.csv'
isFile = os.path.isfile(path_times_train) and os.path.isfile(path_times_test)
#print(isFile)

if not isFile :
    utils.split_times(tmin,tmax,model_number)
    
train_times = pd.read_csv(path_times_train).drop(columns=['Unnamed: 0']).to_numpy().transpose()[0]
test_times = pd.read_csv(path_times_test).drop(columns=['Unnamed: 0']).to_numpy().transpose()[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_train, output_train, input_test, output_test = utils.make_train_test_ds(coarse_factors, len_in, train_times, test_times, Directory)
ins = [input_train, input_test]
outs = [output_train, output_test]

for j in range(len(ins)):
    input = ins[j]
    input = input.reshape(-1,len(variables)-1,nz)
    for i in range(len(variables)-1):
        input[:,i] -= torch.mean(input[:,i])
        input[:,i] /= torch.std(input[:,i])
    input = input.reshape(-1,len_in)
    ins[j] = input

for i in range(len(outs)):
    output = outs[i]
    output -= torch.mean(output)
    output /= torch.std(output)
    outs[i] = output

#----------------MODELS----------------

model_names = ["simple", "PCA", "Conv", "VAE"]
net_params = [[], [], [], []]

for i in range(len(model_names)) :
    name = model_names[i]
    if name == "Conv" :
        ins[1] = ins[1].reshape(-1,len(variables)-1,nz)
    
    model = name(net_params[i])
    model = utils.load_model('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu'))
    model.eval()
    # prediction
    print(ins[1].shape)
    output_pred = model(ins[1])
    # compute loss
    loss = F.mse_loss(output_pred, outs[1], reduction='mean')
