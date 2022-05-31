import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent)
os.chdir(Path(sys.path[0]))
import modules.utils as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna
from optuna.trial import TrialState

print(os.getcwd())
print('cuda available : ', torch.cuda.is_available())


def define_lin_layers(trial, input_features, output_features):
    # We optimize the number of linear layers, hidden units and dropout ratio in each layer.
    n_lins = trial.suggest_int("n_layers", 1, 5)
    layers = []

    in_features = input_features

    for i in range(n_lins):
        out_features = trial.suggest_int("n_units_l{}".format(i), 16, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, output_features))

    return nn.Sequential(*layers)

class CNN(nn.Module):
    def __init__(self, trial, input_features, output_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=0, dilation=1, groups=input_features, bias=True)
        self.conv2 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.bn1 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.regression = define_lin_layers(trial, int(input_features*(output_features-1)/(3*5)), output_features)

        self.input_shape = int(input_features*(output_features-1)/(3*5))
        self.output_shape = output_features
                                        

    def forward(self, x):       # x is of shape (batch_size, input_features, nz), in_size = nz*input_features
        x = F.max_pool1d(input=self.conv1(self.bn1(x)), kernel_size=5)
        x = F.max_pool1d(input=self.conv2(self.bn2(x)), kernel_size=3)
        x = torch.flatten(x, start_dim=1,end_dim=-1)
        return self.regression(x)


def test(model, device, input_test, output_test):
    model.eval()
    # prediction
    output_pred = model(input_test.to(device))
    # compute loss
    test_loss = F.mse_loss(output_pred, output_test.to(device), reduction='mean')
    return test_loss.item()

def train(device, trial, batch_size, nb_epochs, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):

    # define model
    n_batches = input_train.shape[0]//batch_size
    model = CNN(trial, input_features=len_in, output_features=len_out)
    model = model.to(device)

    # Generate the optimizers.
    decay = trial.suggest_float("decay", 0.9, 0.99,)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay, last_epoch= -1)

    for epoch in trange(nb_epochs, leave=False):
        model.train()
        tot_losses=0
        indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
        for i_batch in indexes_arr:
            input_batch = input_train[i_batch,:,:].to(device)
            output_batch = output_train[i_batch,:].to(device)
            optimizer.zero_grad()
            # forward pass
            #print('input_batch device : ', input_batch.get_device())
            #print('output_batch device : ', output_batch.get_device())
            output_pred = model(input_batch)
            # compute loss
            #print("out size :", output_pred.shape, ", out batch size :", output_batch.shape)
            loss = F.mse_loss(output_pred, output_batch, reduction='mean')
            tot_losses += loss.item()
            # backward pass
            loss.backward()
            optimizer.step()
        train_losses.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
        #print(tot_losses)                               # comme on a des batch 2 fois plus petit (16 au lieu de 32)
                                                        # on a une loss en moyenne 2 fois plus petite

        test_losses.append(test(model, device, input_test, output_test))

        if epoch < 100:
            scheduler.step()

    print('Model {},{},{},Epoch [{}/{}], Loss: {:.6f}'.format(lr, decay, batch_size, epoch+1, nb_epochs, tot_losses/n_batches))
    return test_losses[-1]

def objective(trial):
    coarse_factors = [32]
    Directory = f"data"

    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    nz=376

    len_samples = nz*len(variables)
    len_in = nz*(len(variables)-1)
    len_out = nz
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
    ins = [input_train, input_test]
    outs = [output_train, output_test]

    for j in range(len(ins)):
        input = ins[j]
        input = input.reshape(-1,len(variables)-1,nz)
        for i in range(len(variables)-1):
            input[:,i] -= torch.mean(input[:,i])
            input[:,i] /= torch.std(input[:,i])
        ins[j] = input

    for i in range(len(outs)):
        output = outs[i]
        output -= torch.mean(output)
        output /= torch.std(output)
        outs[i] = output

    batch_size = 32             
    nb_epochs = 50            
    train_losses=[]
    test_losses=[]

    obj = train(device, trial, batch_size, nb_epochs, train_losses, test_losses, ins[0], outs[0], ins[1], outs[1], n_in_features, nz)
    return obj

if __name__ == '__main__':

    print("starting optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=10800)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))