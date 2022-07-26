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

import sklearn.metrics as metrics


#----------------MODEL DEF----------------

#simple net model
class DNN(nn.Module):
    def __init__(self, input_size, output_size, drop_prob1=0.2, drop_prob2=0.3, drop_prob3=0.4, hidden_size1=1024, hidden_size2=512, hidden_size3=256):
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

    
    def forward(self, x):
        return self.regression(x)

# pca net model 
class PCA(nn.Module):
    def __init__(self, input_size, output_size, drop_prob1=0.2, drop_prob2=0.3, drop_prob3=0.4, hidden_size1=256, hidden_size2=512, hidden_size3=256):
        super(PCA, self).__init__()
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

# AE model
class AE(nn.Module):
    def __init__(self, input_features=376,  hidden_size1=256, hidden_size2=128, z_dim=3, 
                drop_enc1=0.3, drop_enc2=0.2, drop_enc3=0.3, 
                drop_dec1 = 0.3, drop_dec2=0.2, drop_dec3=0.2):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc1),
                                        nn.Linear(input_features, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc2),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc3),
                                        nn.Linear(hidden_size2, z_dim))
        self.decoder = nn.Sequential(nn.BatchNorm1d(z_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_dec1),
                                        nn.Linear(z_dim, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_dec2),
                                        nn.Linear(hidden_size2, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_dec3),
                                        nn.Linear(hidden_size1, input_features))

        
    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class AE_DNN(nn.Module):
    def __init__(self, u_ae, v_ae, w_ae, theta_ae, s_ae, tke_ae, nn_net):
        super(AE_DNN, self).__init__()
        self.u_ae = u_ae
        self.v_ae = v_ae
        self.w_ae = w_ae
        self.theta_ae = theta_ae
        self.s_ae = s_ae
        self.tke_ae = tke_ae
        self.nn_net = nn_net

    def forward(self, x):
        if x.shape[0] == 6:
            z_u = self.u_ae.encode(x[0])
            z_v = self.v_ae.encode(x[1])
            z_w = self.w_ae.encode(x[2])
            z_theta = self.theta_ae.encode(x[3])
            z_s = self.s_ae.encode(x[4])
            z_tke = self.tke_ae.encode(x[5])
        else :
            z_u = self.u_ae.encode(x[:,0])
            z_v = self.v_ae.encode(x[:,1])
            z_w = self.w_ae.encode(x[:,2])
            z_theta = self.theta_ae.encode(x[:,3])
            z_s = self.s_ae.encode(x[:,4])
            z_tke = self.tke_ae.encode(x[:,5])
        latent_var = torch.cat((z_u, z_v, z_w, z_theta, z_s, z_tke), dim=1)
        return self.nn_net(latent_var)


class AE_CNN(nn.Module):
    def __init__(self, input_features, output_features, drop_prob1=0.301, drop_prob2=0.121, drop_prob3=0.125, hidden_size1=288, hidden_size2=471, hidden_size3=300):
        super(AE_CNN, self).__init__()
        self.encoder = nn.Sequential(nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=0, dilation=1, groups=input_features, bias=True),
                                        nn.MaxPool1d(kernel_size=5, stride=5),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.MaxPool1d(kernel_size=5, stride=5),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.MaxPool1d(kernel_size=3, stride=3))

        self.decoder = nn.Sequential(nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Upsample(scale_factor=3, mode='linear'),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Upsample(scale_factor=5, mode='linear'),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Upsample(scale_factor=5, mode='linear'),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=1, dilation=1, groups=input_features, bias=True))
        
        self.in_pred = int(input_features*(output_features-1)/(5*5*3))
        self.regression = nn.Sequential(nn.BatchNorm1d(self.in_pred, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Linear(self.in_pred, hidden_size1),
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
                                        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):       # x is of shape (batch_size, input_features, nz), in_size = nz*input_features
        x = self.encode(x)
        x = torch.flatten(x, start_dim=1,end_dim=-1)
        return self.regression(x)

class AE_CNN_D(nn.Module):
    def __init__(self, input_features, output_features, drop_prob1=0.053, drop_prob2=0.009, drop_prob3=0.094, drop_prob4=0.209, hidden_size1=117, hidden_size2=458, hidden_size3=255):
        super(AE_CNN_D, self).__init__()
        self.encoder = nn.Sequential(nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=0, dilation=1, groups=input_features, bias=True),
                                        nn.MaxPool1d(kernel_size=5, stride=5),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.MaxPool1d(kernel_size=5, stride=5),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.MaxPool1d(kernel_size=3, stride=3))

        self.decoder = nn.Sequential(nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Upsample(scale_factor=3, mode='linear'),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Upsample(scale_factor=5, mode='linear'),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                        nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Upsample(scale_factor=5, mode='linear'),
                                        nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=1, dilation=1, groups=input_features, bias=True))
        
        self.in_pred = int(input_features*(output_features-1)/(5*5*3))
        self.regression = nn.Sequential(nn.BatchNorm1d(self.in_pred, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Linear(self.in_pred, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob1),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob2),
                                        nn.Linear(hidden_size2, hidden_size3),
                                        nn.ReLU())
        self.mean = nn.Sequential(nn.BatchNorm1d(hidden_size3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob3),
                                        nn.Linear(hidden_size3, output_features))
        self.logvar = nn.Sequential(nn.BatchNorm1d(hidden_size3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob4),
                                        nn.Linear(hidden_size3, output_features))
                                        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):       # x is of shape (batch_size, input_features, nz), in_size = nz*input_features
        x = self.encode(x)
        x = torch.flatten(x, start_dim=1,end_dim=-1)
        return self.mean(self.regression(x))


def main():

    #----------------PREPARE DATA----------------

    coarse_factors = [16,32,64]
    Directory = f"data"

    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    nz=376

    len_in = nz*(len(variables)-1)
    len_out = nz
    latent_dim = 11
    z_dim = 18
    reduced_len = 30
    n_in_features = len(variables)-1

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
    ins = [input_train.to(device), input_test.to(device)]
    outs = [output_train.to(device), output_test.to(device)]

    for j in range(len(ins)):
        input = ins[j]
        input = input.reshape(-1,len(variables)-1,nz)
        for i in range(len(variables)-1):
            input[:,i] -= torch.mean(input[:,i])
            input[:,i] /= torch.std(input[:,i])
        ins[j] = input

    mean_out = torch.mean(outs[0]).cpu().numpy()
    std_out = torch.std(outs[0]).cpu().numpy()

    for i in range(len(outs)):
        output = outs[i]
        output -= torch.mean(output)
        output /= torch.std(output)
        outs[i] = output

    #----------------BASELINE----------------

    baseline_heat_flux = ins[1][:,2,:]*ins[1][:,3,:]
    print("baseline shape :", baseline_heat_flux.shape)
    print("out shape :", outs[1].shape)

    #----------------PARAMS----------------
    t=0
    z=50
    L=32
    largeur=int(512/L)

    #----------------MODEL PREDS----------------

    model_names = ["conv", "pca", "simple", "vae", 'ae', 'conv_ae', 'conv_ae_distrib', 'multiL', 'multiL_d', 'meta_d']
    net_params = [[n_in_features, len_out], [reduced_len, len_out], [len_in,len_out], [len_in,latent_dim,len_out], [z_dim, len_out], [n_in_features, len_out]]
    net_preds = []
    losses = []
    R2s = []

    true_ds = outs[1][t*largeur**2:(t+1)*largeur**2,:].cpu().detach().numpy()

    for i in range(len(model_names)) :
        input_pred = ins[1]
        name = model_names[i]
        print('processing model : ',name)
        if name == "conv" :
            model = CNN(input_features=net_params[i][0] ,output_features=net_params[i][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'pca':
            input_pred = input_pred.reshape(-1,len_in)
            _,_,V = torch.pca_lowrank(torch.concat((ins[0].reshape(-1,len_in), ins[1].reshape(-1,len_in)), axis=0), q=reduced_len)
            input_pred = torch.mm(ins[1].reshape(-1,len_in), V)
            model = PCA(input_size=net_params[i][0] ,output_size=net_params[i][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'simple':
            input_pred = input_pred.reshape(-1,len_in)
            model = DNN(input_size=net_params[i][0] ,output_size=net_params[i][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'vae':
            input_pred = input_pred.reshape(-1,len_in)
            model = VAE(input_features=net_params[i][0], z_dim=net_params[i][1], output_features=net_params[i][2])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'ae':
            nn_model = DNN(input_size=net_params[i][0], output_size=net_params[i][1], hidden_size1=256)
            ae_models = []
            for var in variables[:-1] :
                ae_model = AE(input_features=nz)
                ae_model.load_state_dict(torch.load('explo/models/{}_ae_net.pt'.format(var), map_location=torch.device(device)))
                ae_model.to(device)
                ae_model.eval()
                ae_models.append(ae_model)
            model = AE_DNN(ae_models[0], ae_models[1], ae_models[2], ae_models[3], ae_models[4], ae_models[5], nn_model)
            model.load_state_dict(torch.load('explo/models/dnn_ae_net.pt', map_location=torch.device('cpu')))

        elif name == 'conv_ae':
            model = AE_CNN(input_features=net_params[i][0] ,output_features=net_params[i][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'conv_ae_distrib':
            model = AE_CNN_D(input_features=net_params[i-1][0] ,output_features=net_params[i-1][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'multiL':
            model = AE_CNN_D(input_features=net_params[i-2][0] ,output_features=net_params[i-2][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'multiL_d':
            model = AE_CNN_D(input_features=net_params[i-3][0] ,output_features=net_params[i-3][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        elif name == 'meta_d':
            model = AE_CNN_D(input_features=net_params[i-4][0] ,output_features=net_params[i-4][1])
            model.load_state_dict(torch.load('explo/models/{}_net.pt'.format(name), map_location=torch.device('cpu')))

        else :
            raise Exception("{} model is not supported".format(name))

        model.to(device)
        model.eval()

        # prediction
        output_pred = model(input_pred)
        net_preds.append(output_pred)

        # compute loss
        loss = F.mse_loss(output_pred, outs[1], reduction='mean')

        # compute R^2
        r2 = metrics.r2_score(outs[1].cpu().detach().numpy().reshape(-1,1), output_pred.cpu().detach().numpy().reshape(-1,1))

        losses.append(loss)
        R2s.append(r2)
        print("{} loss : {}".format(name, loss))
        print("{} r2 : {}".format(name, r2))

    '''for i in range(len(model_names)) :
        pred_ds = net_preds[i][t*largeur**2:(t+1)*largeur**2,:].cpu().detach().numpy()
        utils.plot_output(pred_ds,true_ds,L,z,'explo/images/eval/{}_net.png'.format(model_names[i]), color='RdBu_r')'''
    
    #----------------BASELINE MODEL----------------

    #baseline_heat_flux = utils.plot_baseline(Directory, test_times, len_out, z, t, L, mean_out, std_out)
    baseline_loss = F.mse_loss(baseline_heat_flux, outs[1], reduction='mean')
    baseline_r2 = metrics.r2_score(outs[1].cpu().detach().numpy().reshape(-1,1), baseline_heat_flux.cpu().detach().numpy().reshape(-1,1))
    print("baseline loss : {}".format(baseline_loss))
    print("baseline r2 : {}".format(baseline_r2))

    '''
    #----------------Loss v. Hori div----------------

    input_pred = input_pred.reshape(-1,len(variables)-1,nz)
    model_div = AE_CNN(input_features=net_params[i][0] ,output_features=net_params[i][1])
    model_div.load_state_dict(torch.load('explo/models/conv_ae_net.pt', map_location=torch.device('cpu')))

    utils.plot_loss_div(input_pred, outs[1], model_div, L,'explo/images/eval/loss_div.png')
    '''


    


if __name__ == '__main__':
    print("entering main")
    main()