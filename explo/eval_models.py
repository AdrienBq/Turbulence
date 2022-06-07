import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import netCDF4 as nc

import os
import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)
os.chdir(Path(sys.path[0]))
import modules.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F


#----------------MODEL DEF----------------

#simple and pca net model 
class DNN(nn.Module):
    def __init__(self, batch_size, input_size, output_size, drop_prob1=0.2, drop_prob2=0.3, drop_prob3=0.4, hidden_size1=1024, hidden_size2=512, hidden_size3=256):
        super(DNN, self).__init__()
        self.regression = nn.Sequential(nn.BatchNorm1d(input_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Linear(input_size, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob1),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob2),
                                        nn.Linear(hidden_size2, hidden_size3),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob3),
                                        nn.Linear(hidden_size3, output_size)
                                        )
        self.drop_prob1 = drop_prob1
        self.drop_prob2 = drop_prob2
        self.drop_prob3 = drop_prob3
        self.batch_size = batch_size
        self.input_shape = input_size
        self.output_shape = output_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3

    
    def forward(self, x):
        return self.regression(x)

# conv model
class CNN(nn.Module):
    def __init__(self, input_features, output_features, drop_prob1=0.301, drop_prob2=0.121, drop_prob3=0.125, hidden_size1=288, hidden_size2=471, hidden_size3=300):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=0, dilation=1, groups=input_features, bias=True)
        self.conv2 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.bn1 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.regression = nn.Sequential(nn.BatchNorm1d(int(input_features*(output_features-1)/(3*5)), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Linear(int(input_features*(output_features-1)/(3*5)), hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob1),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob2),
                                        nn.Linear(hidden_size2, hidden_size3),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob3),
                                        nn.Linear(hidden_size3, output_features))

        self.drop_prob1 = drop_prob1
        self.drop_prob2 = drop_prob2
        self.drop_prob3 = drop_prob3
        self.input_shape = int(input_features*(output_features-1)/(3*5))
        self.output_shape = output_features
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
                                        

    def forward(self, x):       # x is of shape (batch_size, input_features, nz), in_size = nz*input_features
        x = F.max_pool1d(input=self.conv1(self.bn1(x)), kernel_size=5)
        x = F.max_pool1d(input=self.conv2(self.bn2(x)), kernel_size=3)
        x = torch.flatten(x, start_dim=1,end_dim=-1)
        return self.regression(x)

# VAE model
class VAE(nn.Module):
    def __init__(self, input_features=2256, output_features=376, h_dec_dim=359, h_enc_dim=283, z_dim=11, drop_prob1=0.1080863832594497, drop_prob2=0.1398954543731306, drop_prob3=0.176256881097202, drop_prob4=0.19467179508996357, drop_prob5=0.19408057406110021, hidden_size1=217, hidden_size2=262, hidden_size3=490, hidden_size4=358, hidden_size5=321):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_features, h_enc_dim)
        self.fc2 = nn.Linear(h_enc_dim, z_dim)
        self.fc3 = nn.Linear(h_enc_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dec_dim)
        self.fc5 = nn.Linear(h_dec_dim, input_features)
        self.bn1 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(h_enc_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(h_enc_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = nn.BatchNorm1d(z_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn5 = nn.BatchNorm1d(h_dec_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.d1 = nn.Dropout(p=0.11130569507243004)
        self.d2 = nn.Dropout(p=0.12705405941207876)
        self.dmu = nn.Dropout(p=0.15014610591260366)
        self.dlog_var = nn.Dropout(p=0.44688800536582435)

        self.regression = nn.Sequential(nn.BatchNorm1d(z_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Linear(z_dim, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob1),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob2),
                                        nn.Linear(hidden_size2, hidden_size3),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob3),
                                        nn.Linear(hidden_size3, hidden_size4),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob4),
                                        nn.Linear(hidden_size4, hidden_size5),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob5),
                                        nn.Linear(hidden_size5, output_features)
                                        )

        
    def encode(self, x):
        h = F.relu(self.fc1(self.d1(self.bn1(x))))
        return self.fc2(self.dmu(self.bn2(h))), self.fc3(self.dlog_var(self.bn3(h)))
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(self.d2(self.bn4(z))))
        return self.fc5(self.bn5(h))
    
    def forward_vae(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
    
    def forward(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.regression(z)


def main():

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

    mean_out = torch.mean(outs[0])
    std_out = torch.std(outs[0])

    for i in range(len(outs)):
        output = outs[i]
        output -= torch.mean(output)
        output /= torch.std(output)
        outs[i] = output

    #----------------PARAMS----------------
    t=0
    z=50
    L=32
    largeur=int(512/L)

    #----------------MODEL PREDS----------------

    model_names = ["simple", "pca", "conv", "vae"]
    net_params = [[], [], [], []]
    net_preds = []
    losses = []

    true_ds = outs[1][t*largeur**2:(t+1)*largeur**2,:].cpu().detach().numpy()

    for i in range(len(model_names)) :
        name = model_names[i]
        if name == "conv" :
            ins[1] = ins[1].reshape(-1,len(variables)-1,nz)
            model = CNN(net_params[i])
            utils.load_model('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu'))

        elif name == 'pca' or name == 'simple':
            model = DNN(net_params[i])
            utils.load_model('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu'))
        else :
            model = VAE(net_params[i])
            model = utils.load_model('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu'))
            model.eval()

        model.eval()

        # prediction
        print(ins[1].shape)
        output_pred = model(ins[1])
        net_preds.append(output_pred)

        # compute loss
        losses.append(F.mse_loss(output_pred, outs[1], reduction='mean'))

    for i in range(len(model_names)) :
        pred_ds = net_preds[i][t*largeur**2:(t+1)*largeur**2,:].cpu().detach().numpy()
        utils.plot_output(pred_ds,true_ds,L,z,'explo/images/{}_net.png'.format(model_names[i]), color='RdBu_r')


    #----------------BASELINE MODEL----------------

    baseline_heat_flux = utils.plot_baseline(Directory, test_times, len_out, z, t, L, mean_out, std_out)


if __name__ == '__main__':
    print("entering main")
    main()