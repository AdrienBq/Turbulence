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
    The network outputs a mean and a variance instead of a single value to model the incertainty of the prediction.
    The convolutional auto_encoder network with 3 convolutional layers for the encoder and 4 for the decoder. 
    Latent space dim is 6*5=30.
    Uses batchnorm and max pooling layers between convolutional layers and batchnorm, dropout and relu activation functions for linear layers.
    Feedforward net has 4 fully connected layers using batchnorm, dropout and ReLU activation.
    '''
    def __init__(self, input_features, output_features, drop_prob1=0.053, drop_prob2=0.009, drop_prob3=0.094, drop_prob4=0.209, hidden_size1=117, hidden_size2=458, hidden_size3=255):
        '''
        ## Description
        Double neural network combining an AE with a simple feedforward network.
        The AE maps the input to a latent space and the feedforward network maps the latent space to the output to predict fluxes.
        The network outputs a mean and a variance instead of a single value to model the incertainty of the prediction.
        The convolutional auto_encoder network with 3 convolutional layers for the encoder and 4 for the decoder. 
        Latent space dim is 6*5=30.
        Uses batchnorm and max pooling layers between convolutional layers and batchnorm, dropout and relu activation functions for linear layers.
        Feedforward net has 4 fully connected layers using batchnorm, dropout and ReLU activation.
        Parameters are optimized using Optuna in the conv_ae_net_optuna.py file.

        ## Parameters
        - input_features (int) : number of input features (number of input channels of the first convolutional layer)
        - output_features (int) : number of output features (size of the output of the last linear layer)
        - drop_prob1 (float) : dropout probability for the first hidden layer of the feedforward net, default : 0.053
        - drop_prob2 (float) : dropout probability for the second hidden layer, default : 0.009
        - drop_prob3 (float) : dropout probability for the third hidden layer, default : 0.094
        - drop_prob4 (float) : dropout probability for the fourth hidden layer, default : 0.209 
        - hidden_size1 (int) : number of neurons in the first hidden layer, default : 227 
        - hidden_size2 (int) : number of neurons in the second hidden layer, default : 458
        - hidden_size3 (int) : number of neurons in the third hidden layer, default : 255
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
        #return self.mean(self.regression(x))

def custom_loss(mu, logvar, obj):
    var = torch.exp(logvar)
    var2 = torch.mul(var,var)
    div = torch.divide(torch.mul(mu-obj,mu-obj),(2*var2))
    return torch.sum(logvar + div)


def test(model, device, input_test, output_test, l_factors):
    '''
    ## Description
    Test the meta-model on the test set.

    ## Parameters
    - model (torch.nn.Module) : the model to test
    - device (torch.device) : the device to use (cpu / gpu)
    - input_test (list of torch.tensor) : the input test set. Each list element corresponds to a different coarse grained level.
    - output_test (torch.tensor) : the output test set.
    '''
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
        tot_loss += (ae_loss + test_loss)*l_factors[l]

    return tot_loss.item(), ae_loss.item(), log_lik.item(), test_loss.item()

def train(device, batch_size, nb_epochs, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, len_out):
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
    - input_train (list of torch.Tensor) : the training input data. Each list element corresponds to a different coarse grained level.
    - output_train (list of torch.Tensor) : the training output data. Each list element corresponds to a different coarse grained level.
    - input_test (list of torch.Tensor) : the test input data. Each list element corresponds to a different coarse grained level.
    - output_test (list of torch.Tensor) : the test output data. Each list element corresponds to a different coarse grained level.
    - len_in (int) : the length of the input data (here it's the number of input channels of the first convolutional layer)
    - len_out (int) : the length of the output data
    '''

    n_batches = [input_train[i].shape[0]//batch_size for i in range(len(input_train))]
    meta_model = AE_CNN(input_features=len_in,output_features=len_out)
    meta_model = meta_model.to(device)

    meta_lr = 5.6*1e-3
    meta_decay = 0.92

    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, meta_decay, last_epoch= -1)

    l_factors = [1,4,16]

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
        test_loss = test(meta_model, device, input_test, output_test, l_factors)
        test_losses.append(test_loss)

        if epoch < 40 :
            meta_scheduler.step()

        if epoch%2 == 0:
            print('ae_loss :', test_loss[1], 'log-likelihood :', test_loss[2], 'pred_loss :', test_loss[3])

    print('Model : meta_lr [{}], meta_decay [{:.4f}], Epoch [{}/{}], ae_loss: {:.6f}, pred_loss : {:.6f}'.format(meta_lr, meta_decay, epoch+1, nb_epochs, test_losses[-1][1],test_losses[-1][3]))
    
    return meta_model



def main():
    '''
    ## Description
    main function : create the datasets, train and test the models, save and plot the results
    '''
    coarse_factors = [16,32,64]
    largeurs = [int(512//coarse_factor) for coarse_factor in coarse_factors]
    Directory = f"data"

    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    nz=376      #output size (vertical size of the simulation)

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

    input_train, output_train, input_test, output_test = utils.make_train_test_ds(coarse_factors, len_in, train_times, test_times, Directory)
    #print(input_train.shape)

    in_train_list = [input_train[sum(len(train_times)*largeurs[i]**2 for i in range(j)):
                    sum(len(train_times)*largeurs[i]**2 for i in range(j+1))] for j in range(len(largeurs)-1)]
    in_train_list.insert(0,input_train[:len(train_times)*largeurs[0]**2])
    #print(len(in_train_list), in_train_list[0].shape, in_train_list[1].shape, in_train_list[2].shape)

    in_test_list = [input_test[sum(len(test_times)*largeurs[i]**2 for i in range(j)):
                    sum(len(test_times)*largeurs[i]**2 for i in range(j+1))] for j in range(len(largeurs)-1)]
    in_test_list.insert(0,input_test[:len(test_times)*largeurs[0]**2])

    out_train_list = [output_train[sum(len(train_times)*largeurs[i]**2 for i in range(j)): 
                    sum(len(train_times)*largeurs[i]**2 for i in range(j+1))] for j in range(len(largeurs)-1)]
    out_train_list.insert(0,output_train[:len(train_times)*largeurs[0]**2])

    out_test_list = [output_test[sum(len(test_times)*largeurs[i]**2 for i in range(j)):
                    sum(len(test_times)*largeurs[i]**2 for i in range(j+1))] for j in range(len(largeurs)-1)]
    out_test_list.insert(0,output_test[:len(test_times)*largeurs[0]**2])

    ins = [in_train_list, in_test_list]
    outs = [out_train_list, out_test_list]

    for k in range(len(ins)):
        for j in range(len(ins[k])):
            input = ins[k][j]
            input = input.reshape(-1,len(variables)-1,nz)
            for i in range(len(variables)-1):
                input[:,i] -= torch.mean(input[:,i])
                input[:,i] /= torch.std(input[:,i])
            ins[k][j] = input

    for k in range(len(outs)):
        for j in range(len(outs)):
            output = outs[k][j]
            output -= torch.mean(output)
            output /= torch.std(output)
            outs[k][j] = output

    batch_size = 32             # obligé de le mettre à 16 si pls L car sinon le nombre total de samples n'est pas divisible par batch_size 
    nb_epochs = 100              # et on ne peut donc pas reshape. Sinon il ne pas prendre certains samples pour que ça tombe juste.
    train_losses=[]
    test_losses=[]

    meta_model = train(device, batch_size, nb_epochs, train_losses, test_losses, ins[0], outs[0], ins[1], outs[1], n_in_features, nz)
    train_losses_arr = np.array(train_losses)
    test_losses_arr = np.array(test_losses)

    torch.save(meta_model.state_dict(), f"explo/models/multiL_d_net_2.pt")

    try :
        plt.plot(train_losses_arr[1:], label='train loss')
        plt.plot(test_losses_arr[1:], label='test pred loss')
        plt.title(f"AE CONV net training")
        plt.legend()
        plt.show()
        plt.savefig(f"explo/images/16_64_d_loss.png")
    except :
        pass


if __name__ == '__main__':
    main()