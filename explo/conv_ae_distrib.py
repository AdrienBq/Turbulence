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


class AE_CNN(nn.Module):
    '''
    ## Description
    Double neural network combining an AE with a simple feedforward network.
    The AE maps the input to a latent space and the feedforward network maps the latent space to the output to predict fluxes.
    The convolutional auto_encoder network with 3 convolutional layers for the encoder and 4 for the decoder. 
    Latent space dim is 6*5=30.
    Uses batchnorm and max pooling layers between convolutional layers and batchnorm, dropout and relu activation functions for linear layers.
    Feedforward net has 4 fully connected layers using batchnorm, dropout and ReLU activation.
    '''
    def __init__(self, input_features, output_features, drop_prob1=0.301, drop_prob2=0.121, drop_prob3=0.125, drop_prob4=0.125, hidden_size1=288, hidden_size2=471, hidden_size3=300):
        '''
        ## Description
        Double neural network combining an AE with a simple feedforward network.
        The AE maps the input to a latent space and the feedforward network maps the latent space to the output to predict fluxes.
        The convolutional auto_encoder network with 3 convolutional layers for the encoder and 4 for the decoder. 
        Latent space dim is 6*5=30.
        Uses batchnorm and max pooling layers between convolutional layers and batchnorm, dropout and relu activation functions for linear layers.
        Feedforward net has 4 fully connected layers using batchnorm, dropout and ReLU activation.
        Parameters are optimized using Optuna in the conv_ae_net_optuna.py file.

        ## Parameters
        - input_features (int) : number of input features (number of input channels of the first convolutional layer)
        - output_features (int) : number of output features (size of the output of the last linear layer)
        - drop_prob1 (float) : dropout probability for the first hidden layer of the feedforward net, default : 0.301
        - drop_prob2 (float) : dropout probability for the second hidden layer, default : 0.121
        - drop_prob3 (float) : dropout probability for the third hidden layer, default : 0.125
        - hidden_size1 (int) : number of neurons in the first hidden layer, default : 288
        - hidden_size2 (int) : number of neurons in the second hidden layer, default : 471
        - hidden_size3 (int) : number of neurons in the third hidden layer, default : 300
        '''
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

    def loss(mu, logvar, x):
        var = torch.exp(logvar)
        return torch.sum(logvar + torch.divide(torch.mul(mu-x,mu-x)/(2*torch.mul(var,var))))

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
    test_loss = model.loss(output_pred[0], output_test.to(device), reduction='mean')
    tot_loss = ae_loss + test_loss
    return tot_loss.item(), ae_loss.item(), test_loss.item()

def train(device, batch_size, nb_epochs, models, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):
    '''
    ## Description
    Train the model on the training set. Loop to train multiple models with different learning rates, batch sizes and decays.

    ## Parameters
    - device: the device to use for the model
    - batch_sizes (int) : list of batch sizes to use
    - nb_epochs (int) : number of epochs to train the model
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
    
    n_batches = input_train.shape[0]//batch_size
    model = AE_CNN(input_features=len_in,output_features=len_out)
    model = model.to(device)

    lr_enc = 5.03*1e-3
    decay_enc = 0.909
    lr_dec = 5.96*1e-3
    decay_dec = 0.968
    lr_reg = 1.09*1e-3 
    decay_reg = 0.933

    optimizer_enc = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc)
    scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(optimizer_enc, decay_enc, last_epoch= -1)
    optimizer_dec = torch.optim.Adam(model.decoder.parameters(), lr=lr_dec)
    scheduler_dec = torch.optim.lr_scheduler.ExponentialLR(optimizer_dec, decay_dec, last_epoch= -1)
    optimizer_reg = torch.optim.Adam(model.regression.parameters(), lr=lr_reg)
    scheduler_reg = torch.optim.lr_scheduler.ExponentialLR(optimizer_reg, decay_reg, last_epoch= -1)
    optimizer_mean = torch.optim.Adam(model.mean.parameters(), lr=lr_reg)
    scheduler_mean = torch.optim.lr_scheduler.ExponentialLR(optimizer_mean, decay_reg, last_epoch= -1)
    optimizer_logvar = torch.optim.Adam(model.logvar.parameters(), lr=lr_reg)
    scheduler_logvar = torch.optim.lr_scheduler.ExponentialLR(optimizer_logvar, decay_reg, last_epoch= -1)

    for epoch in trange(nb_epochs, leave=False):
        model.train()
        for param in model.regression.parameters():
            param.requires_grad = False
        if epoch>30 :
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
            optimizer_mean.zero_grad()
            optimizer_logvar.zero_grad()
            # forward pass
            output_ae = model.decode(model.encode(input_batch))
            mu,logvar = model(input_batch)
            # compute loss
            ae_loss = F.mse_loss(output_ae,input_batch, reduction='mean')
            pred_loss = model.loss(mu,logvar,output_batch)
            loss = ae_loss + pred_loss
            tot_losses += pred_loss.item()
            # backward pass
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            optimizer_reg.step()
            optimizer_mean.step()
            optimizer_logvar.step()

        train_losses.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
        test_loss = test(model, device, input_test, output_test)
        test_losses.append(test_loss)

        if epoch%10 == 0:
            print('ae_loss :', test_loss[1], 'pred_loss :', test_loss[2])

        if epoch < 100:
            scheduler_enc.step()
            scheduler_dec.step()
            if epoch>30:
                scheduler_reg.step()
                scheduler_mean.step()
                scheduler_logvar.step()

    models.append(model)
    print('Model : lr_enc [{}], decay_enc [{:.4f}], lr_dec[{}], decay_dec [{:.4f}], lr_reg [{}], decay_reg [{:.4f}], Epoch [{}/{}], ae_loss: {:.6f}, pred_loss : {:.6f}'.format(lr_enc, decay_enc, lr_dec, decay_dec, lr_reg, decay_reg, epoch+1, nb_epochs, test_losses[-1][1],test_losses[-1][2]))
                


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

    batch_size = 32             # obligé de le mettre à 16 si pls L car sinon le nombre total de samples n'est pas divisible par batch_size 
    nb_epochs = 70               # et on ne peut donc pas reshape. Sinon il ne pas prendre certains samples pour que ça tombe juste.
    train_losses=[]
    test_losses=[]
    models=[]

    train(device, batch_size, nb_epochs, models, train_losses, test_losses, ins[0], outs[0], ins[1], outs[1], n_in_features, nz)
    train_losses_arr = np.array(train_losses)
    test_losses_arr = np.array(test_losses)

    torch.save(models[0].state_dict(), f"explo/models/conv_ae_net_opt.pt")

    try :
        plt.plot(train_losses_arr[1:], label='train')
        plt.plot(test_losses_arr[:], label='test')
        plt.title(f"AE CONV net training")
        plt.legend()
        plt.show()
        plt.savefig(f"explo/images/losses_conv_ae.png")
    except :
        pass



if __name__ == '__main__':
    main()