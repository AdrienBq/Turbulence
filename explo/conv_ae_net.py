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


class CNN(nn.Module):
    '''
    ## Description
    Convolutional neural network with 2 convolutional layers and 4 fully connected layers. 
    Uses batchnorm and max pooling layers between convolutional layers and batchnorm, dropout and relu activation functions for linear layers.
    '''
    def __init__(self, input_features, output_features, drop_prob1=0.301, drop_prob2=0.121, drop_prob3=0.125, hidden_size1=288, hidden_size2=471, hidden_size3=300):
        '''
        ## Description
        Convolutional neural network with 2 convolutional layers and 4 fully connected layers. 
        Uses batchnorm and max pooling layers between convolutional layers and batchnorm, dropout and relu activation functions for linear layers.
        Parameters are optimized using Optuna in the conv_net_optuna.py file.

        ## Parameters
        - input_features (int) : number of input features (number of input channels of the first convolutional layer)
        - output_features (int) : number of output features (size of the output of the last linear layer)
        - drop_prob1 (float) : dropout probability for the first hidden layer, default : 0.301
        - drop_prob2 (float) : dropout probability for the second hidden layer, default : 0.121
        - drop_prob3 (float) : dropout probability for the third hidden layer, default : 0.125
        - hidden_size1 (int) : number of neurons in the first hidden layer, default : 288
        - hidden_size2 (int) : number of neurons in the second hidden layer, default : 471
        - hidden_size3 (int) : number of neurons in the third hidden layer, default : 300
        '''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=0, dilation=1, groups=input_features, bias=True)
        self.conv2 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.conv3 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.bn1 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.deconv1 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.deconv2 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.deconv3 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=3, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.deconv4 = nn.Conv1d(in_channels=input_features, out_channels=input_features, kernel_size=2, stride=1, padding=1, dilation=1, groups=input_features, bias=True)
        self.bn4 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn5 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn6 = nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
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
        x = F.max_pool1d(input=self.conv1(self.bn1(x)), kernel_size=5)
        x = F.max_pool1d(input=self.conv2(self.bn2(x)), kernel_size=5)
        x = F.max_pool1d(input=self.conv3(self.bn3(x)), kernel_size=3)
        return x

    def decode(self, x):
        x = F.upsample(input=self.bn4(self.deconv1(x)), scale_factor=3, mode='linear')
        x = F.upsample(input=self.bn5(self.deconv2(x)), scale_factor=5, mode='linear')
        x = F.upsample(input=self.bn6(self.deconv3(x)), scale_factor=5, mode='linear')
        return self.deconv4(x)

    def forward(self, x):       # x is of shape (batch_size, input_features, nz), in_size = nz*input_features
        x = self.encode(x)
        x = torch.flatten(x, start_dim=1,end_dim=-1)
        return self.regression(x)

def test(model, device, input_test, output_test):
    '''
    ## Description
    Test the model on the test set.

    ## Parameters
    - model (torch.nn.Module) : the model to test
    - device (torch.device) : the device to use (cpu / gpu)
    - input_test (torch.tensor) : the input test set
    - output_test (torch.tensor) : the output test set
    '''
    model.eval()
    # prediction
    ae_output = model.decode(model.encode(input_test.to(device)))
    output_pred = model(input_test.to(device))
    # compute loss
    ae_loss = F.mse_loss(ae_output, input_test.to(device))
    test_loss = F.mse_loss(output_pred, output_test.to(device), reduction='mean')
    tot_loss = ae_loss + test_loss
    return tot_loss.item(), ae_loss.item(), test_loss.item()

def train(device, learning_rates, decays, batch_sizes, nb_epochs, models, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):
    '''
    ## Description
    Train the model on the training set. Loop to train multiple models with different learning rates, batch sizes and decays.

    ## Parameters
    - device: the device to use for the model
    - learning_rates (list) : list of learning rates to use
    - decays (list) : list of decays to use
    - batch_sizes (list) : list of batch sizes to use
    - nb_epochs (list) : number of epochs to train the model
    - models (list) : empty list to store the models
    - train_losses (list) : empty list to store the training losses
    - test_losses (list) : empty list to store the test losses
    - input_train (torch.Tensor) : the training input data
    - output_train (torch.Tensor) : the training output data
    - input_test (torch.Tensor) : the test input data
    - output_test (torch.Tensor) : the test output data
    - len_in (int) : the length of the input data (here it's the number of input channels of the first convolutional layer)
    - len_out (int) : the length of the output data
    '''
    for learning_rate in learning_rates:
        train_losses_lr = []
        test_losses_lr = []
        for decay in decays:
            train_losses_decay = []
            test_losses_decay = []
            for batch_size in batch_sizes :
                n_batches = input_train.shape[0]//batch_size
                model = CNN(input_features=len_in,output_features=len_out)
                model = model.to(device)
                print(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay, last_epoch= -1)
                models.append(model)
                train_losses_bs = []
                test_losses_bs = []
                for epoch in trange(nb_epochs[0], leave=False):
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
                        optimizer.zero_grad()
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
                        optimizer.step()
                    train_losses_bs.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
                    #print(tot_losses)                               # comme on a des batch 2 fois plus petit (16 au lieu de 32)
                                                                    # on a une loss en moyenne 2 fois plus petite
                    test_loss = test(model, device, input_test, output_test)
                    test_losses_bs.append(test_loss[0])

                    if epoch%10 == 0:
                        print('ae_loss :', test_loss[1], 'pred_loss :', test_loss[2])

                    if epoch < 100:
                        scheduler.step()

                print('Model {},{},{},Epoch [{}/{}], ae_loss: {:.6f}, pred_loss : {:.6f}'.format(learning_rate, decay, batch_size, epoch+1, nb_epochs[0], test_losses_bs[-1][1], test_losses_bs[-1][2]))
                train_losses_decay.append(train_losses_bs[0])
                test_losses_decay.append(test_losses_bs[0])
            train_losses_lr.append(train_losses_decay)
            test_losses_lr.append(test_losses_decay)
        train_losses.append(train_losses_lr)
        test_losses.append(test_losses_lr)


def main():
    '''
    ## Description
    main function : create the datasets, train and test the models, save and plot the results
    '''
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

    learning_rates = [1e-3, 1e-4]
    decays = [0.99, 0.97, 0.95]
    batch_sizes = [32]             # obligé de le mettre à 16 si pls L car sinon le nombre total de samples n'est pas divisible par batch_size 
    nb_epochs = [50]               # et on ne peut donc pas reshape. Sinon il ne pas prendre certains samples pour que ça tombe juste.
    train_losses=[]
    test_losses=[]
    models=[]

    train(device, learning_rates, decays, batch_sizes, nb_epochs, models, train_losses, test_losses, ins[0], outs[0], ins[1], outs[1], n_in_features, nz)
    train_losses_arr = np.array(train_losses)
    test_losses_arr = np.array(test_losses)

    torch.save(models[0].state_dict(), f"explo/models/conv_net_opt_{0}.pt")

    fig,axes = plt.subplots(len(learning_rates),len(batch_sizes)*len(decays),figsize=(5*len(learning_rates),4*len(batch_sizes)*len(decays)))

    for i in range(len(learning_rates)):
        for j in range(len(decays)):
            for k in range(len(batch_sizes)):
                try : 
                    axes[i,k+j*len(batch_sizes)].plot(train_losses_arr[i,j,k,1:], label='train')
                    axes[i,k+j*len(batch_sizes)].plot(test_losses_arr[i,j,k,1:], label='test')
                    axes[i,k+j*len(batch_sizes)].set_title(f"d = {decays[j]}, lr = {learning_rates[i]}, bs = {batch_sizes[k]}")
                    axes[i,k+j*len(batch_sizes)].legend()
                except :
                    pass
    try :
        axes.plot(train_losses_arr[0,0,0,1:], label='train')
        axes.plot(test_losses_arr[0,0,0,1:], label='test')
        axes.set_title(f"d = {decays[0]}, lr = {learning_rates[0]}, bs = {batch_sizes[0]}")
        axes.legend()
    except :
        pass

    plt.show()
    plt.savefig(f"explo/images/losses_conv_ae.png")


if __name__ == '__main__':
    main()