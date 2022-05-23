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

print(os.getcwd())
print('cuda available : ', torch.cuda.is_available())


class DNN(nn.Module):
    def __init__(self, batch_size, input_size, output_size, drop_prob1=0.2, drop_prob2=0.3, drop_prob3=0.4, hidden_size1=256, hidden_size2=512, hidden_size3=256):
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

def test(model, device, input_test, output_test):
    model.eval()                                    # on a une loss en moyenne 2 fois plus petite
    # prediction
    output_pred = model(input_test.to(device))
    # compute loss
    test_loss = F.mse_loss(output_pred, output_test.to(device), reduction='mean')
    return test_loss.item()

def train(device, learning_rates, decays, batch_sizes, nb_epochs, models, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):
    for learning_rate in learning_rates:
        train_losses_lr = []
        test_losses_lr = []
        for decay in decays:
            train_losses_decay = []
            test_losses_decay = []
            for batch_size in batch_sizes :
                n_batches = input_train.shape[0]//batch_size
                model = DNN(batch_size=batch_size,input_size=len_in,output_size=len_out)
                model = model.to(device)
                print(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay, last_epoch= -1)
                models.append(model)
                train_losses_bs = []
                test_losses_bs = []
                for epoch in trange(nb_epochs[0], leave=False):
                    model.train()
                    tot_losses=0
                    indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
                    for i_batch in indexes_arr:
                        input_batch = input_train[i_batch,:].to(device)
                        output_batch = output_train[i_batch,:].to(device)
                        optimizer.zero_grad()
                        # forward pass
                        #print('input_batch device : ', input_batch.get_device())
                        #print('output_batch device : ', output_batch.get_device())
                        output_pred = model(input_batch)
                        # compute loss
                        loss = F.mse_loss(output_pred, output_batch, reduction='mean')
                        tot_losses += loss.item()
                        # backward pass
                        loss.backward()
                        optimizer.step()
                    train_losses_bs.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
                    #print(tot_losses)                               # comme on a des batch 2 fois plus petit (16 au lieu de 32)
                                                                    # on a une loss en moyenne 2 fois plus petite

                    test_losses_bs.append(test(model, device, input_test, output_test))

                    if epoch < 300:
                        scheduler.step()

                print('Model {},{},{},Epoch [{}/{}], Loss: {:.6f}'.format(learning_rate, decay, batch_size, epoch+1, nb_epochs[0], tot_losses/n_batches))
                train_losses_decay.append(train_losses_bs)
                test_losses_decay.append(test_losses_bs)
            train_losses_lr.append(train_losses_decay)
            test_losses_lr.append(test_losses_decay)
        train_losses.append(train_losses_lr)
        test_losses.append(test_losses_lr)

def main():
    coarse_factors = [32]
    Directory = "data"

    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    nz=376

    len_samples = nz*len(variables)
    len_in = nz*(len(variables)-1)
    len_out = nz
    reduced_len = 30

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

    input_train, output_train, input_test, output_test = utils.make_train_test_ds(coarse_factors, len_in, train_times, test_times, Directory)
    ins = [input_train, input_test]
    outs = [output_train, output_test]

    for j in range(len(ins)):
        input = ins[j]
        input = input.reshape(-1,len(variables)-1,nz)
        for i in range(len(variables)-1):
            input[:,i] -= input[:,i].mean()
            input[:,i] /= input[:,i].std()
        input = input.reshape(-1,(len(variables)-1)*nz)
        input = input.to(device)
        ins[j] = input

    for i in range(len(outs)):
        output = outs[i]
        output -= output.mean()
        output /= output.std()
        output = output.to(device)
        outs[i] = output

    U,S,V = torch.pca_lowrank(torch.concat((ins[0], ins[1]), axis=0), q=reduced_len)

    for i in range(len(ins)) :
        ins[i] = torch.mm(ins[i], V)

    learning_rates = [5*1e-3]
    decays = [0.97,0.95,0.93]
    batch_sizes = [32]             # obligé de le mettre à 16 si pls L car sinon le nombre total de samples n'est pas divisible par batch_size 
    nb_epochs = [75]   # et on ne peut donc pas reshape. Sinon il ne pas prendre certains samples pour que ça tombe juste.
    train_losses=[]
    test_losses=[]
    models=[]

    train(device, learning_rates, decays, batch_sizes, nb_epochs, models, train_losses, test_losses, ins[0], outs[0], ins[1], outs[1], reduced_len, len_out)

    #for i in range(len(models)):
        #torch.save(models[i].state_dict(), f"explo/models/pca_{i}.pt")

    fig,axes = plt.subplots(len(decays),len(batch_sizes)*len(learning_rates),figsize=(5*len(decays),4*len(batch_sizes)*len(learning_rates)))

    for i in range(len(learning_rates)):
        for j in range(len(decays)):
            for k in range(len(batch_sizes)):
                axes[j,k+i*len(batch_sizes)].plot(train_losses[i][j][k][2:], label='train')
                axes[j,k+i*len(batch_sizes)].plot(test_losses[i][j][k][2:], label='test')
                axes[j,k+i*len(batch_sizes)].set_title(f"d = {decays[i]}, lr = {learning_rates[j]}, bs = {batch_sizes[k]}")
                axes[j,k+i*len(batch_sizes)].legend()
    plt.show()

    plt.savefig(f"explo/images/losses_pca_3.png")


if __name__ == '__main__':
    main()