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


class AE_CNN(nn.Module):
    def __init__(self, input_features, output_features, drop_prob1=0.301, drop_prob2=0.121, drop_prob3=0.125, drop_prob4=0.125, hidden_size1=288, hidden_size2=471, hidden_size3=300):
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
        return self.mean(self.regression(x)), self.logvar(self.regression(x))

def custom_loss(mu, logvar, obj):
    var = torch.exp(logvar)
    var2 = torch.mul(var,var)
    div = torch.divide(torch.mul(mu-obj,mu-obj),(2*var2))
    return torch.sum(logvar + div)


def test(model, device, input_test, output_test):
    model.eval()
    ae_loss = 0
    log_lik = 0
    test_loss = 0
    tot_loss = 0

    for l in range(len(input_test)):
        #forward pass
        input = input_test[l].to(device)
        ae_output = model.decode(model.encode(input))
        output_pred = model(input)

        #compute loss
        output = output_test[l].to(device)
        ae_loss += F.mse_loss(ae_output, input, reduction='mean')
        log_lik += custom_loss(output_pred[0], output_pred[1], output)
        test_loss += F.mse_loss(output_pred[0], output, reduction='mean')
        tot_loss += ae_loss + test_loss

    return tot_loss.item(), ae_loss.item(), log_lik.item(), test_loss.item()

def train(device, trial, batch_size, nb_epochs, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):

    n_batches = [input_train[i].shape[0]//batch_size for i in range(len(input_train))]
    meta_model = AE_CNN(input_features=len_in,output_features=len_out)
    meta_model = meta_model.to(device)

    meta_lr = trial.suggest_float("meta_lr", 1e-5, 1e-2, log=True)
    meta_decay = trial.suggest_float("meta_decay", 1e-5, 1e-2, log=True)

    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, meta_decay, last_epoch= -1)

    l_factors = [trial.suggest_int("l_factor_"+str(i), 1, 10) for i in range(len(input_train))]

    for epoch in trange(nb_epochs, leave=False):
        tot_losses=0
        indexes = []
        for l in range(len(input_train)):
            indexes.append(np.random.permutation(input_train[l].shape[0]).reshape(-1, batch_size))

        #outer loop :
        for i in range(indexes[-1].shape[0]):
            meta_optimizer.zero_grad()
            meta_model.train()

            #inner loop : train the local models
            for l in range(len(input_train)):
                nb_batches_l = n_batches[l]//min(n_batches)
            
                for j in range(nb_batches_l):
                    i_batch = indexes[l][i*nb_batches_l+j]
                    input_batch = input_train[l][i_batch,:,:].to(device)
                    output_batch = output_train[l][i_batch,:].to(device)

                    # forward pass
                    output_ae = meta_model.decode(meta_model.encode(input_batch))
                    mu,logvar = meta_model(input_batch)
                    
                    # compute loss
                    ae_loss = F.mse_loss(output_ae,input_batch, reduction='mean')
                    pred_loss = F.mse_loss(mu,output_batch)
                    log_lik = custom_loss(mu, logvar, output_batch)
                    loss = ae_loss + pred_loss
                    tot_losses += l_factors[l]*loss.item()

                    # backward pass
                    loss.backward()

            meta_optimizer.step()

        train_losses.append(tot_losses/sum(n_batches[i] for i in range(len(input_train))))     # loss moyenne sur tous les batchs 
        test_loss = test(meta_model, device, input_test, output_test)
        test_losses.append(test_loss)

        meta_scheduler.step()

        trial.report(test_losses[-1][0], epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    print('Model : lr_enc [{:.4f}], decay_enc [{:.4f}], lr_dec[{:.4f}], decay_dec [{:.4f}], lr_reg [{:.4f}], decay_reg [{:.4f}], Epoch [{}/{}], ae_loss: {:.6f}, pred_loss : {:.6f}'.format(lr_enc, decay_enc, lr_dec, decay_dec, lr_reg, decay_reg, epoch+1, nb_epochs, test_losses[-1][1],test_losses[-1][3]))
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
    nb_epochs = 0      
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