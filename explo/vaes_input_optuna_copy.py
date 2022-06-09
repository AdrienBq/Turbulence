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


def define_net_layers(trial, net, input_features, output_features):
    # We optimize the number of linear layers, hidden units and dropout ratio in each layer.
    n_lins = trial.suggest_int("n_layers", 1, 5)
    layers = []
    mu_layer = []
    logvar_layer = []

    in_features = input_features

    for i in range(n_lins):
        out_features = trial.suggest_int("n_{}_units_l{}".format(net,i), 64, 512)
        layers.append(nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        p = trial.suggest_float("{}_dropout_l{}".format(net,i), 0.1, 0.5)
        layers.append(nn.Dropout(p))
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features

    if net == 'enc':
        mu_layer.append(nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        p_mu = trial.suggest_float("mu_dropout", 0.1, 0.5)
        mu_layer.append(nn.Dropout(p_mu))
        mu_layer.append(nn.Linear(in_features, output_features))

        logvar_layer.append(nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        p_logvar = trial.suggest_float("sig_dropout", 0.1, 0.5)
        logvar_layer.append(nn.Dropout(p_logvar))
        logvar_layer.append(nn.Linear(in_features, output_features))
    
    else :
        layers.append(nn.BatchNorm1d(in_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        p = trial.suggest_float("{}_dropout_l{}".format(net,i), 0.1, 0.5)
        layers.append(nn.Dropout(p))
        layers.append(nn.Linear(in_features, output_features))
        
    return nn.Sequential(*layers), nn.Sequential(*mu_layer), nn.Sequential(*logvar_layer)


class VAE(nn.Module):
    def __init__(self, trial, input_features, latent_features):
        super(VAE, self).__init__()
        self.bulk_encoder, self.mu_layer, self.sig_layer = define_net_layers(trial, "enc", input_features, latent_features)
        self.decoder = define_net_layers(trial, "dec", latent_features, input_features)[0]

        self.input_shape = input_features
        self.latent_shape = latent_features
                                        
    def encode(self, x):
        x = self.bulk_encoder(x)
        mu = self.mu_layer(x)
        logvar = self.sig_layer(x)
        return mu, logvar
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def test(model, device, input_test):
    model.eval()
    # prediction
    input_batch = input_test[:,j,:].to(device)
    x_reconst, mu, log_var = model(input_batch)
    # compute loss
    reconst_loss = F.mse_loss(x_reconst, input_batch, reduction='mean')
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    test_loss =  reconst_loss + kl_div
    return test_loss.item()

def train(device, trial, batch_size, nb_epochs, train_losses, test_losses, input_train, input_test, len_in):

    latent_dim = trial.suggest_int("latent_dim", 2, 10)
    # define model
    n_batches = input_train.shape[0]//batch_size
    model_vae = VAE(trial, input_features=len_in, latent_features=latent_dim)
    model_vae = model_vae.to(device)

    # Generate the optimizers.
    decay_vae = trial.suggest_float("decay_vae", 0.9, 0.99,)
    #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer_name = "Adam"
    lr_vae = trial.suggest_float("lr_vae", 1e-5, 1e-2, log=True)
    optimizer_vae = getattr(optim, optimizer_name)(model_vae.parameters(), lr=lr_vae)

    optimizer_vae = torch.optim.Adam(model_vae.parameters(), lr=lr_vae)
    scheduler_vae = torch.optim.lr_scheduler.ExponentialLR(optimizer_vae, decay_vae, last_epoch= -1)

    for epoch in trange(nb_epochs, leave=False):
        tot_losses=0
        indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
        for i_batch in indexes_arr:
            model_vae.train()
            input_batch = input_train[i_batch].to(device)
            # forward pass
            x_reconst, mu, log_var = model_vae(input_batch)

            # compute loss
            reconst_loss = F.mse_loss(x_reconst, input_batch, reduction='mean')
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss =  reconst_loss + kl_div
            tot_losses += loss.item()

            # backward pass
            loss.backward()
            optimizer_vae.step()

        train_losses.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
        #print(tot_losses)                               # comme on a des batch 2 fois plus petit (16 au lieu de 32)
                                                        # on a une loss en moyenne 2 fois plus petite

        test_losses.append(test(model_vae, device, input_test))

        if epoch < 100:
            scheduler_vae.step()

        trial.report(test_losses[-1], epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    #print('Model {},{},{},Epoch [{}/{}], Train Loss: {:.6f}'.format(lr, decay, batch_size, epoch+1, nb_epochs, tot_losses/n_batches))
    return test_losses[-1]

def objective(trial):
    coarse_factors = [32]
    Directory = f"data"

    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    var = 0
    nz=376

    full_len_in = nz*(len(variables)-1)
    n_in_features = len(variables)-1
    len_in = nz

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

    input_train, _, input_test, _ = utils.make_train_test_ds(coarse_factors, full_len_in, train_times, test_times, Directory)
    ins = [input_train.reshape(-1,len(variables)-1,nz), input_test.reshape(-1,len(variables)-1,nz)]

    for j in range(len(ins)):
        input = ins[j][:,var,:]
        for i in range(len(variables)-1):
            input[:,i] -= torch.mean(input[:,i])
            input[:,i] /= torch.std(input[:,i])
        ins[j] = input

    batch_size = 32
    nb_epochs = 10
    train_losses=[]
    test_losses=[]

    obj = train(device, trial, batch_size, nb_epochs, train_losses, test_losses, ins[0], ins[1], len_in)
    return obj

if __name__ == '__main__':

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
    print("starting optimization")
    study.optimize(objective, n_trials=30, timeout=10800)

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