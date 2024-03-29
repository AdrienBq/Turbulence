import pandas as pd
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path

sys.path[0] = str(Path(sys.path[0]).parent.parent)
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
        out_features = trial.suggest_int("n_units_l{}".format(i), 64, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, output_features))

    return nn.Sequential(*layers)

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


def test(model, device, input_test, output_test):
    model.eval()
    # prediction
    ae_output = model.decode(model.encode(input_test.to(device)))
    output_pred = model(input_test.to(device))
    # compute loss
    ae_loss = F.mse_loss(ae_output, input_test.to(device))
    test_loss = F.mse_loss(output_pred, output_test.to(device), reduction='mean')
    tot_loss = ae_loss + test_loss
    return tot_loss.item(), ae_loss.item(), test_loss.item()

def train(device, trial, batch_size, nb_epochs, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):

    n_batches = input_train.shape[0]//batch_size
    model = AE_CNN(input_features=len_in,output_features=len_out)
    model = model.to(device)

    lr_enc = trial.suggest_float("lr_enc", 1e-5, 1e-2, log=True)
    decay_enc = trial.suggest_float("decay_enc", 0.9, 0.99,)
    lr_dec = trial.suggest_float("lr_dec", 1e-5, 1e-2, log=True)
    decay_dec = trial.suggest_float("decay_dec", 0.9, 0.99,)
    lr_reg = trial.suggest_float("lr_reg", 1e-5, 1e-2, log=True)
    decay_reg = trial.suggest_float("decay_reg", 0.9, 0.99,)

    optimizer_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc)
    scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_enc, decay_enc, last_epoch= -1)
    optimizer_dec = torch.optim.Adam(model.decoder.parameters(), lr=lr_dec)
    scheduler_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_dec, decay_dec, last_epoch= -1)
    optimizer_reg = torch.optim.Adam(model.regression.parameters(), lr=lr_reg)
    scheduler_reg = torch.optim.lr_scheduler.ExponentialLR(optimizer_reg, decay_reg, last_epoch= -1)

    for epoch in trange(nb_epochs, leave=False):
        model.train()
        for param in model.regression.parameters():
            param.requires_grad = False
        if epoch>20 :
            for param in model.regression.parameters():
                param.requires_grad = True
        tot_losses=0
        indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
        for i_batch in indexes_arr:
            input_batch = input_train[i_batch,:,:].to(device)
            output_batch = output_train[i_batch,:].to(device)
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            optimizer_reg.zero_grad()
            # forward pass
            output_ae = model.decode(model.encode(input_batch))
            output_pred = model(input_batch)
            # compute loss
            ae_loss = F.mse_loss(output_ae, input_batch, reduction='mean')
            pred_loss = F.mse_loss(output_pred, output_batch, reduction='mean')
            loss = ae_loss + pred_loss
            tot_losses += pred_loss.item()
            # backward pass
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            optimizer_reg.step()

        train_losses.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
        test_loss = test(model, device, input_test, output_test)
        test_losses.append(test_loss)

        if epoch < 100:
            scheduler_enc.step()
            scheduler_dec.step()
            if epoch>20:
                scheduler_reg.step()

        trial.report(test_losses[-1][0], epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    print('Model : lr_enc [{:.4f}], decay_enc [{:.4f}], lr_dec[{:.4f}], decay_dec [{:.4f}], lr_reg [{:.4f}], decay_reg [{:.4f}], Epoch [{}/{}], ae_loss: {:.6f}, pred_loss : {:.6f}'.format(lr_enc, decay_enc, lr_dec, decay_dec, lr_reg, decay_reg, epoch+1, nb_epochs, test_losses[-1][1],test_losses[-1][2]))
    return test_losses[-1][0]

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

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
    print("starting optimization")
    print('using cuda : ', torch.cuda.is_available())
    study.optimize(objective, n_trials=50, timeout=10800)

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