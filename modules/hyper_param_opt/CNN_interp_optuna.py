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


def test(model, device, coarse_input_test, input_test):
    model.eval()
    # prediction
    coarse_test = np.zeros((input_test.shape[0], input_test.shape[1], 256))
    for var in range(input_test.shape[1]):
        for sample in range(input_test.shape[0]):
            coarse_test[sample,var,:] = utils.interpolation_linear(coarse_input_test[sample,var,:], N_output=256)
    coarse_test = torch.from_numpy(coarse_test).to(device)
    output_pred = model(coarse_test)
    # compute loss
    test_loss = F.mse_loss(output_pred, input_test.to(device), reduction='mean')
    return test_loss.item()

def train(device, trial, batch_size, nb_epochs, train_losses, test_losses, coarse_input_train, input_train, coarse_input_test, input_test, len_in, len_out):

    lr_conv = trial.suggest_float("lr_enc", 1e-5, 1e-2, log=True)
    decay_conv = trial.suggest_float("decay_enc", 0.9, 0.99,)
    lr_reg = trial.suggest_float("lr_reg", 1e-5, 1e-2, log=True)
    decay_reg = trial.suggest_float("decay_reg", 0.9, 0.99,)

    drop_prob1=trial.suggest_uniform("drop_prob1", 0.0, 0.5)
    drop_prob2=trial.suggest_uniform("drop_prob2", 0.0, 0.5)
    hidden_size1=trial.suggest_int("hidden_size1", 100, 500)
    hidden_size2=trial.suggest_int("hidden_size2", 100, 500)

    n_batches = input_train.shape[0]//batch_size
    model = CNN(input_features=len_in,output_features=6*len_out, drop_prob1=drop_prob1, drop_prob2=drop_prob2, hidden_size1=hidden_size1, hidden_size2=hidden_size2).to(device)
    model = model.to(device)

    optimizer_conv = torch.optim.Adam(model.conv.parameters(), lr=lr_conv)
    scheduler_conv = torch.optim.lr_scheduler.ExponentialLR(optimizer_conv, decay_conv, last_epoch= -1)
    optimizer_reg = torch.optim.Adam(model.regression.parameters(), lr=lr_reg)
    scheduler_reg = torch.optim.lr_scheduler.ExponentialLR(optimizer_reg, decay_reg, last_epoch= -1)

    for epoch in trange(nb_epochs, leave=False):
        model.train()
        tot_losses=0
        indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
        for idx in trange(indexes_arr.shape[0]):
            i_batch = indexes_arr[idx,:]
            coarse_input_batch = coarse_input_train[i_batch,:,:]
            coarse_interp = np.zeros((batch_size, coarse_input_train.shape[1], 256))
            for var in range(coarse_input_batch.shape[1]):
                for sample in range(batch_size):
                    #print(coarse_input_batch[sample,var,:].shape)
                    coarse_interp[sample,var,:] = utils.interpolation_cubic(coarse_input_batch[sample,var,:], N_output=256)
            coarse_interp = torch.from_numpy(coarse_interp).to(device)
            input_batch = input_train[i_batch,:,:].to(device)
            optimizer_conv.zero_grad()
            optimizer_reg.zero_grad()
            # forward pass
            output_pred = model(coarse_interp)
            # compute loss
            pred_loss = F.mse_loss(output_pred, input_batch, reduction='mean')
            loss = pred_loss
            tot_losses += pred_loss.item()
            # backward pass
            loss.backward()
            optimizer_conv.step()
            optimizer_reg.step()

        train_losses.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
        test_loss = test(model, device,coarse_input_test,input_test)
        test_losses.append(test_loss)

        if epoch%5 == 0:
            print('loss :', test_loss)

        if epoch < 100:
            scheduler_conv.step()
            scheduler_reg.step()                

        trial.report(test_losses[-1], epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    print('Model : lr_conv [{}], decay_conv [{:.4f}], lr_reg [{}], decay_reg [{:.4f}], Epoch [{}/{}], pred_loss : {:.6f}'.format(lr_conv, decay_conv, lr_reg, decay_reg, epoch+1, nb_epochs,test_losses[-1]))

    return test_losses[-1]

def objective(trial):
    coarse_factors = [32]
    Directory = f"data"

    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    nz=376
    len_in = nz*(len(variables)-1)
    n_in_features = len(variables)-1

    model_number = 11
    tmin=1
    tmax=62+1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using cuda : ', torch.cuda.is_available())

    path_times_train = f'data/test_train_times/times_train_{model_number}.csv'
    path_times_test = f'data/test_train_times/times_test_{model_number}.csv'
    isFile = os.path.isfile(path_times_train) and os.path.isfile(path_times_test)
    #print(isFile)

    if not isFile :
        utils.split_times(tmin,tmax,model_number)
        
    train_times = pd.read_csv(path_times_train).drop(columns=['Unnamed: 0']).to_numpy().transpose()[0]
    test_times = pd.read_csv(path_times_test).drop(columns=['Unnamed: 0']).to_numpy().transpose()[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_train, _, input_test, _ = utils.make_train_test_ds(coarse_factors, len_in, train_times, test_times, Directory)
    ins = [input_train, input_test]
    coarses = []

    for j in range(len(ins)):
        input = ins[j]
        input = input.reshape(-1,len(variables)-1,nz)
        for i in range(len(variables)-1):
            input[:,i] -= torch.mean(input[:,i])
            input[:,i] /= torch.std(input[:,i])
        ins[j] = input

        coarse = np.zeros((input.shape[0],input.shape[1],input.shape[2]//2))
        for i in range(input.shape[2]//2):
            coarse[:,:,i] = input[:,:,2*i]
        coarses.append(coarse)

    batch_size = 32             
    nb_epochs = 10      
    train_losses=[]
    test_losses=[]

    obj = train(device, trial, batch_size, nb_epochs, train_losses, test_losses, coarses[0], ins[0], coarses[1], ins[1], n_in_features, nz)
    return obj

if __name__ == '__main__':

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
    print("starting optimization")
    print('using cuda : ', torch.cuda.is_available())
    study.optimize(objective, n_trials=15, timeout=10800)

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