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

# AE model
class AE(nn.Module):
    '''
    ## Description
    AE neural network for the inputs
    The point is to reduce the dimension of the input by using a VAE. 
    The AE maps the input to a latent space and later a feedforward network will predict fluxes from the encode input in the latent space.
    '''
    def __init__(self, input_features=376,  hidden_size1=256, hidden_size2=128, z_dim=3, 
                drop_enc1=0.3, drop_enc2=0.2, drop_enc3=0.3, 
                drop_dec1 = 0.3, drop_dec2=0.2, drop_dec3=0.2):
        '''
        ## Description
        AE neural network for the inputs
        The point is to reduce the dimension of the input by using a VAE. 
        The AE maps the input to a latent space and later a feedforward network will predict fluxes from the encode input in the latent space.
        The hyperparameters were optimized using the vae_dnn_net_optuna.py script. They have to be changed for each input.
        The AE uses fully connected layers with batchnorm, dropout and relu activation functions. It has 2 hidden layer for the encoder and 2 hidden layer for the decoder.
        
        ## Parameters
        - input_features: number of input features (input of the VAE)
        - hidden_size1: number of hidden units in the encoder first hidden layer and decoder second hidden layer, default=256
        - hidden_size2: number of hidden units in the encoder second hidden layer and decoder third hidden layer, default=128
        - z_dim: dimension of the latent space, default=3
        - drop_enc1: dropout probability in the encoder first hidden layer, default=0.3
        - drop_enc2: dropout probability in the encoder second hidden layer, default=0.2
        - drop_mu: dropout probability in the mu layer, default=0.3
        - drop_log_var: dropout probability in the log_var layer, default=0.25
        - drop_dec1: dropout probability in the decoder first hidden layer, default=0.3
        - drop_dec2: dropout probability in the decoder second hidden layer, default=0.2
        - drop_dec3: dropout probability in the decoder third hidden layer, default=0.2
        '''
        super(AE, self).__init__()
        self.encoder = nn.Sequential(nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc1),
                                        nn.Linear(input_features, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc2),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc3),
                                        nn.Linear(hidden_size2, z_dim))
        self.decoder = nn.Sequential(nn.BatchNorm1d(z_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_dec1),
                                        nn.Linear(z_dim, hidden_size2),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_dec2),
                                        nn.Linear(hidden_size2, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_dec3),
                                        nn.Linear(hidden_size1, input_features))

        
    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class DNN(nn.Module):
    '''
    ## Description
    Simple feed forward network with 3 hidden layers using linear layers, batchnorm, dropout and ReLU activation. All layers are fully connected.
    '''
    def __init__(self, input_size, output_size, drop_prob1=0.2, drop_prob2=0.3, drop_prob3=0.4, hidden_size1=256, hidden_size2=512, hidden_size3=256):
        '''
        ## Description
        initialise a simple feed forward network with 3 hidden layers using linear layers, batchnorm, dropout and ReLU activation. All lyaers are fully connected.

        ## Parameters
        - input_size (int) : number of input features
        - output_size (int) : number of output features
        - drop_prob1 (float) : dropout probability for the first hidden layer, default : 0.2
        - drop_prob2 (float) : dropout probability for the second hidden layer, default : 0.3
        - drop_prob3 (float) : dropout probability for the third hidden layer, default : 0.4
        - hidden_size1 (int) : number of neurons in the first hidden layer, default : 1024
        - hidden_size2 (int) : number of neurons in the second hidden layer, default : 512
        - hidden_size3 (int) : number of neurons in the third hidden layer, default : 256
        '''
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
    
    def forward(self, x):
        return self.regression(x)

def test(model, device, ae_models, input_test, output_test):
    '''
    ## Description
    test the model on the test set

    ## Parameters
    - model (torch.nn.Module) : the model to test
    - device (torch.device) : the device on which the model is trained
    - input_test (torch.Tensor) : the input test set
    - output_test (torch.Tensor) : the output test set
    '''
    model.eval()                                    # on a une loss en moyenne 2 fois plus petite
    # prediction
    latent_variables = []
    for var in range(len(ae_models)) :
        z = ae_models[var].encode(input_test[:,var,:].to(device)).detach.numpy()
        latent_variables.append(z)
    latent_tensor = torch.from_numpy(np.array(latent_variables)).reshape(-1, z.shape[0]*len(ae_models)).to(device)
    output_pred = model(latent_tensor)
    # compute loss
    test_loss = F.mse_loss(output_pred, output_test.to(device), reduction='mean')
    return test_loss.item()

def train(device, learning_rates, ae_models, nb_epochs, models, train_losses, test_losses, input_train, output_train, input_test, output_test, batch_size, n_batches, len_in, len_out):
    '''
    ## Description
    train the model on the training set. Loop to train different models with different learning rates.

    ## Parameters
    - device (torch.device) : the device on which the model is trained
    - learning_rates (list) : the learning rates to use for the optimizer
    - nb_epochs (list) : the number of epochs to train the model with each learning rate
    - models (list) : empty list to store the models
    - train_losses (list) : empty list to store the training losses
    - test_losses (list) : empty list to store the test losses
    - input_train (torch.Tensor) : the input training set
    - output_train (torch.Tensor) : the output training set
    - input_test (torch.Tensor) : the input test set
    - output_test (torch.Tensor) : the output test set
    - batch_size (int) : the batch size to use for training
    - n_batches (int) : the number of batches to use for training
    - len_in (int) : the length of the input sequence
    - len_out (int) : the length of the output sequence
    '''
    for i in range(len(learning_rates)):
        model = DNN(input_size=len_in,output_size=len_out)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[i])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch= -1)
        for epoch in trange(nb_epochs, leave=False):
            model.train()
            tot_losses=0
            indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
            for i_batch in indexes_arr:
                latent_variables = []
                input_batch = input_train[i_batch].to(device)
                output_batch = output_train[i_batch].to(device)
                optimizer.zero_grad()
                # forward pass
                for var in range(len(ae_models)) :
                    z = ae_models[var].encode(input_batch[:,var,:]).detach().numpy()
                    latent_variables.append(z)
                latent_tensor = torch.from_numpy(np.array(latent_variables)).reshape(-1, z.shape[1]*len(ae_models)).to(device)
                output_pred = model(latent_tensor)
                # compute loss
                loss = F.mse_loss(output_pred, output_batch, reduction='mean')
                tot_losses += loss.item()
                # backward pass
                loss.backward()
                optimizer.step()
            train_losses.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
            #print(tot_losses)                               # comme on a des batch 2 fois plus petit (16 au lieu de 32)
                                                                # on a une loss en moyenne 2 fois plus petite

            test_losses.append(test(model, device, input_test, output_test))

            if epoch < 200:
                scheduler.step()

        models.append(model)
        print('Model {},Epoch [{}/{}], Loss: {:.6f}'.format(i+1,epoch+1, nb_epochs[i], tot_losses/n_batches))

def main():
    '''
    ## Description
    main function : create the datasets, train and test the models, save and plot the results
    '''
    coarse_factors = [64,32,16]
    Directory = "data"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using cuda : ', torch.cuda.is_available())

    nz=376
    latent_dim = 3
    variables=['u', 'v', 'w', 'theta', 's', 'tke', 'wtheta']
    ae_models = []
    for var in variables[:-1]:
        model = AE(input_features=nz)
        model.load_state_dict(torch.load('explo/models/{}_ae_net.pt'.format(var), map_location=torch.device(device)))
        model.to(device)
        model.eval()
        ae_models.append(model)

    nb_var = len(variables) - 1
    full_len_in = nz*(len(variables)-1)
    len_in = nz
    len_out = nz

    model_number = 11
    tmin=1
    tmax=62+1

    path_times_train = f'data/test_train_times/times_train_{model_number}.csv'
    path_times_test = f'data/test_train_times/times_test_{model_number}.csv'
    isFile = os.path.isfile(path_times_train) and os.path.isfile(path_times_test)
    #print(isFile)

    if not isFile :
        utils.split_times(tmin,tmax,model_number)
        
    train_times = pd.read_csv(path_times_train).drop(columns=['Unnamed: 0']).to_numpy().transpose()[0]
    test_times = pd.read_csv(path_times_test).drop(columns=['Unnamed: 0']).to_numpy().transpose()[0]

    input_train, output_train, input_test, output_test = utils.make_train_test_ds(coarse_factors, full_len_in, train_times, test_times, Directory)
    ins = [input_train, input_test]
    outs = [output_train, output_test]

    for j in range(len(ins)):
        input = ins[j].reshape(-1,len(variables)-1,nz)
        for i in range(len(variables)-1):
            input[:,i] -= input[:,i].mean()
            input[:,i] /= input[:,i].std()
        input = input.to(device)
        ins[j] = input

    for j in range(len(outs)):
        output = outs[j]
        output -= output.mean()
        output /= output.std()
        output = output.to(device)
        outs[j] = output
    
    input_train, input_test, output_train, output_test = ins[0], ins[1], outs[0], outs[1]

    learning_rates = [1e-2,1e-3]
    batch_size = 32             # obligé de le mettre à 16 si pls L car sinon le nombre total de samples n'est pas divisible par batch_size 
    nb_epochs = 25   # et on ne peut donc pas reshape. Sinon il ne pas prendre certains samples pour que ça tombe juste.
    train_losses=[]
    test_losses=[]
    models=[]
    n_batches = input_train.shape[0]//batch_size

    train(device, learning_rates, ae_models, nb_epochs, models, train_losses, test_losses, input_train, output_train, input_test, output_test, batch_size, n_batches, nb_var*latent_dim, len_out)

    torch.save(models[0].state_dict(), f"explo/models/dnn_ae_net.pt")

    fig,axes = plt.subplots(1,len(learning_rates),figsize=(20,4))

    for i in range(len(learning_rates)):
        try :
            axes[i].plot(train_losses[i][2:], label="train")
            axes[i].plot(test_losses[i][2:], label="test")
            axes[i].set_title(f"loss (initial lr = {learning_rates[i]}, gamma = 0.95)")
            axes[i].legend()
        except :
            pass
    
    

    plt.savefig(f"explo/images/losses_dnn_ae.png")


if __name__ == '__main__':
    main()