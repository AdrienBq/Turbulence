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


# VAE model
class VAE(nn.Module):
    '''
    ## Description
    VAE neural network for the inputs
    The point is to reduce the dimension of the input by using a VAE. 
    The VAE maps the input to a latent space and later a feedforward network will predict fluxes from the encode input in the latent space.
    '''
    def __init__(self, input_features=2256,  hidden_size1=256, hidden_size2=128, z_dim=3, 
                drop_enc1=0.3, drop_enc2=0.2, drop_mu=0.3, drop_log_var=0.25, 
                drop_dec1 = 0.3, drop_dec2=0.3, drop_dec3=0.3):
        '''
        ## Description
        VAE neural network for the inputs
        The point is to reduce the dimension of the input by using a VAE. 
        The VAE maps the input to a latent space and later a feedforward network will predict fluxes from the encode input in the latent space.
        The hyperparameters were optimized using the vae_dnn_net_optuna.py script. They have to be changed for each input.
        The VAE uses fully connected layers with batchnorm, dropout and relu activation functions. It has 2 hidden layer for the encoder and 2 hidden layer for the decoder.
        
        ## Parameters
        - input_features: number of input features (input of the VAE)
        - hidden_size1: number of hidden units in the encoder first hidden layer and decoder second hidden layer, default=256
        - hidden_size2: number of hidden units in the encoder second hidden layer and decoder third hidden layer, default=128
        - z_dim: dimension of the latent space, default=3
        - drop_enc1: dropout probability in the encoder first hidden layer, default=0.3
        - drop_enc2: dropout probability in the encoder second hidden layer, default=0.2
        - drop_mu: dropout probability in the mu layer, default=0.3
        - drop_log_var: dropout probability in the log_var layer, default=0.25
        '''
        super(VAE, self).__init__()
        self.bulk_encoder = nn.Sequential(nn.BatchNorm1d(input_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc1),
                                        nn.Linear(input_features, hidden_size1),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(hidden_size1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_enc2),
                                        nn.Linear(hidden_size1, hidden_size2),
                                        nn.ReLU())
        self.mu_layer = nn.Sequential(nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.Dropout(drop_mu),
                                        nn.Linear(hidden_size2, z_dim))
        self.log_var_layer = nn.Sequential(nn.BatchNorm1d(hidden_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                            nn.Dropout(drop_log_var),
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
        h = F.relu(self.fc1(self.d1(self.bn1(x))))
        return self.fc2(self.dmu(self.bn2(h))), self.fc3(self.dlog_var(self.bn3(h)))
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(self.d2(self.bn4(z))))
        return self.fc5(self.bn5(h))
    
    def forward_vae(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
    
    def forward(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.regression(z)

def test(model_vae, device, input_test, output_test):
    '''
    ## Description
    Test the model on the test set.

    ## Parameters
    - model (torch.nn.Module) : the model to test
    - device (torch.device) : the device to use (cpu / gpu)
    - input_test (torch.tensor) : the input test set
    - output_test (torch.tensor) : the output test set
    '''
    model_vae.eval()
    # prediction
    output_pred = model_vae(input_test.to(device))
    # compute loss
    test_loss = F.mse_loss(output_pred, output_test.to(device), reduction='mean')
    return test_loss.item()

def train(device, lr_vae, decay_vae, batch_sizes, nb_epochs, models, train_losses, test_losses, input_train, output_train, input_test, output_test, len_in, latent_dim, len_out):
    '''
    ## Description
    Train the model on the training set. Loop to train multiple models with different batch sizes.

    ## Parameters
    - device: the device to use for the model
    - lr_vae: the learning rate for the model
    - decay_vae: the learning rate decay for the model
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
    - latent_dim (int) : the dimension of the latent space
    - len_out (int) : the length of the output data
    '''
    for batch_size in batch_sizes :
        n_batches = input_train.shape[0]//batch_size
        model_vae = VAE(input_features=len_in, z_dim=latent_dim, output_features=len_out)
        model_vae = model_vae.to(device)

        optimizer_vae = torch.optim.Adam(model_vae.parameters(), lr=lr_vae)
        scheduler_vae = torch.optim.lr_scheduler.ExponentialLR(optimizer_vae, decay_vae, last_epoch= -1)
        models.append(model_vae)

        train_losses_bs = []
        test_losses_bs = []
        for epoch in trange(nb_epochs[0], leave=False):
            model_vae.train()
            tot_losses=0
            indexes_arr = np.random.permutation(input_train.shape[0]).reshape(-1, batch_size)
            for i_batch in indexes_arr:
                input_batch = input_train[i_batch].to(device)
                output_batch = output_train[i_batch].to(device)
                optimizer_vae.zero_grad()

                # forward pass
                x_reconst, mu, log_var = model_vae.forward_vae(input_batch)
                output_pred = model_vae(input_batch)

                # compute loss
                reconst_loss = F.mse_loss(x_reconst, input_batch, reduction='sum')
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                flux_pred_loss = F.mse_loss(output_pred, output_batch, reduction='mean')
                loss =  flux_pred_loss + reconst_loss + kl_div
                tot_losses += flux_pred_loss.item()

                # backward pass
                loss.backward()
                optimizer_vae.step()
            train_losses_bs.append(tot_losses/n_batches)     # loss moyenne sur tous les batchs 
            #print(tot_losses)                               # comme on a des batch 2 fois plus petit (16 au lieu de 32)
                                                            # on a une loss en moyenne 2 fois plus petite

            test_losses_bs.append(test(model_vae, device, input_test, output_test))

            if epoch < 50:
                scheduler_vae.step()

        print('Model {},{},{},Epoch [{}/{}], Loss: {:.6f}'.format(lr_vae, decay_vae, batch_size, epoch+1, nb_epochs[0], tot_losses/n_batches))
    train_losses.append(train_losses_bs)
    test_losses.append(test_losses_bs)


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
        input = input.reshape(-1,len_in)
        ins[j] = input

    for i in range(len(outs)):
        output = outs[i]
        output -= torch.mean(output)
        output /= torch.std(output)
        outs[i] = output

    latent_dim = 11
    lr_vae, lr_ff = 0.00155415456328059 , 0.0001095908946314874
    decay_vae, decay_ff = 0.9106719211794353 , 0.9636446685306491
    batch_sizes = [32]             # obligé de le mettre à 16 si pls L car sinon le nombre total de samples n'est pas divisible par batch_size 
    nb_epochs = [150]               # et on ne peut donc pas reshape. Sinon il ne pas prendre certains samples pour que ça tombe juste.
    train_losses=[]
    test_losses=[]
    models=[]

    train(device, lr_vae, decay_vae, batch_sizes, nb_epochs, models, train_losses, test_losses, ins[0], outs[0], ins[1], outs[1], len_in, latent_dim, len_out)
    train_losses_arr = np.array(train_losses)
    test_losses_arr = np.array(test_losses)

    torch.save(models[0].state_dict(), f"explo/models/vae_net_opt_{1}.pt")

    fig,axes = plt.subplots(len(batch_sizes),figsize=(5,4*len(batch_sizes)))

    for k in range(len(batch_sizes)):
        try : 
            axes[k].plot(train_losses_arr[i,j,k,1:], label='train')
            axes[k].plot(test_losses_arr[i,j,k,1:], label='test')
            axes[k].set_title(f"bs = {batch_sizes[k]}")
            axes[k].legend()
        except :
            pass
    try :
        axes.plot(train_losses_arr[0,1:], label='train')
        axes.plot(test_losses_arr[0,1:], label='test')
        axes.set_title(f"d = {decay_vae},{decay_ff}, lr = {lr_vae},{lr_ff}, bs = {batch_sizes[0]}")
        axes.legend()
    except :
        pass

    plt.show()
    plt.savefig(f"explo/images/losses_vae_opt.png")


if __name__ == '__main__':
    main()