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


class CNN(nn.Module):
    def __init__(self, input_features, output_features=6*376, drop_prob1=0.301, drop_prob2=0.121, hidden_size1=288, hidden_size2=471):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                    nn.MaxPool1d(kernel_size=4, stride=4),
                                    nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                    nn.MaxPool1d(kernel_size=4, stride=4),
                                    nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True),
                                    nn.MaxPool1d(kernel_size=4, stride=4))

        self.in_pred = int(input_features*256/(4**3))

        self.regression = nn.Sequential(nn.BatchNorm1d(self.in_pred, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Linear(self.in_pred, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob1),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_prob2),
                                        nn.Linear(hidden_size2, output_features))
                                        

    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1,end_dim=-1)
        return self.regression(x).reshape(-1, 6, 376)


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


def main():

    #----------------PREPARE DATA----------------

    coarse_factors = [32]
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
    coarses = []

    for j in range(len(ins)):
        input = ins[j]
        input = input.reshape(-1,len(variables)-1,nz)
        for i in range(len(variables)-1):
            input[:,i] -= torch.mean(input[:,i])
            input[:,i] /= torch.std(input[:,i])
        ins[j] = input
        coarse = torch.zeros((input.shape[0],input.shape[1],input.shape[2]//2))
        for i in range(input.shape[2]//2):
            coarse[:,:,i] = input[:,:,2*i]
        coarses.append(coarse)

    mean_out = torch.mean(outs[0]).cpu().numpy()
    std_out = torch.std(outs[0]).cpu().numpy()

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

    interp_method = ["linear", "cubic", "knn", 'cnn']
    net_params = [n_in_features, len_out]
    losses = []
    net_preds = []
    true_ds = outs[1][t*largeur**2:(t+1)*largeur**2,:].cpu().detach().numpy()

    for method in interp_method :
        if method == 'cnn' :
            interp_net = CNN(input_features=6).to(device)
            interp_net.load_state_dict(torch.load('explo/models/cnn_interp_net.pt',map_location=torch.device('cpu')))
            interp_net.to(device)
            interp_net.eval()
            coarse_test = torch.zeros((coarses[1].shape[0], coarses[1].shape[1], 256))
            for var in range(coarses[1].shape[1]):
                for sample in range(coarses[1].shape[0]):
                    coarse_test[sample,var,:] = utils.interpolation_linear(coarses[1][sample,var,:], N_output=256)
            coarse_test = coarse_test.to(device)
            input_pred = interp_net(coarse_test)

        elif method == 'knn':
            coarse_test = np.zeros((coarses[1].shape[0], coarses[1].shape[1], 376))
            for var in range(coarses[1].shape[1]):
                for sample in range(coarses[1].shape[0]):
                    coarse_test[sample,var,:] = utils.interpolation_knn(coarses[1][sample,var,:], N_output=376)
            input_pred = torch.from_numpy(coarse_test).to(device)

        elif method == 'linear':
            coarse_test = np.zeros((coarses[1].shape[0], coarses[1].shape[1], 376))
            for var in range(coarses[1].shape[1]):
                for sample in range(coarses[1].shape[0]):
                    coarse_test[sample,var,:] = utils.interpolation_linear(coarses[1][sample,var,:], N_output=376)
            input_pred = torch.from_numpy(coarse_test).to(device)

        elif method == 'cubic':
            coarse_test = np.zeros((coarses[1].shape[0], coarses[1].shape[1], 376))
            for var in range(coarses[1].shape[1]):
                for sample in range(coarses[1].shape[0]):
                    coarse_test[sample,var,:] = utils.interpolation_cubic(coarses[1][sample,var,:], N_output=376)
            input_pred = torch.from_numpy(coarse_test).to(device)
        
        else:
            print('interpolation method not found')
            exit()
        
        model_pred = AE_CNN(input_features=net_params[0] ,output_features=net_params[1])
        model_pred.load_state_dict(torch.load('explo/models/conv_ae_net.pt', map_location=torch.device('cpu'))) 

        model_pred.to(device)
        model_pred.eval()

        # prediction
        print(input_pred.shape)
        input_pred = input_pred.float()
        output_pred = model_pred(input_pred)
        net_preds.append(output_pred)

        # compute loss
        loss = F.mse_loss(output_pred, outs[1], reduction='mean')
        losses.append(loss.item())

        print("{} loss : {}".format(method, loss))

    for i in range(len(interp_method)) :
        pred_ds = net_preds[i][t*largeur**2:(t+1)*largeur**2,:].cpu().detach().numpy()
        utils.plot_output(pred_ds,true_ds,L,z,'explo/images/eval/{}_net.png'.format(interp_method[i]), color='RdBu_r')


if __name__ == '__main__':
    print("entering main")
    main()