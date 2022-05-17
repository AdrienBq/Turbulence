import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from tqdm import tqdm, trange
from netCDF4 import Dataset
import netCDF4 as nc

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

#os.chdir(Path(sys.path[0]).parent)


#--------------PRINT FUNCTIONS-----------------------

def print_one_alt(path_data,var,alt,color='RdBu_r'):
    '''
    ## Description
    print first altitude layer of a data file

    ## Parameters
    - path_data (str) : path to the data file to be plotted
    - color (str) : colormap of the plot
    '''
    nc_init = nc.Dataset(path_data)
    arr = nc_init[f'{var}xy{alt}'][:].filled()[0,0,:,:]
    im = plt.imshow(arr , cmap = color , interpolation = 'nearest' )
    plt.colorbar(im)
    plt.title(f"2-D Heat Map of {var}")
    plt.show()


def plot_output(pred_ds,true_ds,L,z,color='RdBu'):
    '''
    ## Description
    Plot the prediction and true dataset for a given altitude

    ## Parameters
    - pred_ds (np array) : prediction dataset
    - true_ds (np array) : true dataset
    - L (int) : coarsening factor
    - z (int) : altitude index
    - color (str) : color map
    '''
    exp_shapex = int(512/L)
    exp_shapey = int(512/L)
    exp_shapez = 376
    assert pred_ds.shape == true_ds.shape, 'prediction and true datasets have different shapes'
    assert exp_shapex*exp_shapey == pred_ds.shape[0] and exp_shapez == pred_ds.shape[1], 'datasets do not have expected shape'
    
    pred_z = pred_ds[:,z].reshape((exp_shapex,exp_shapey))
    true_z = true_ds[:,z].reshape((exp_shapex,exp_shapey))
    
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 4), ncols=3)
    
    true = ax1.imshow(true_z, cmap=color, interpolation='nearest')
    fig.colorbar(true, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    ax1.set_title(f"2-D Heat Map of the true heat-flux at Altitude {z}")

    pred = ax2.imshow(pred_z, cmap=color, interpolation='nearest')
    fig.colorbar(pred, ax=ax2,orientation='vertical', fraction=0.046, pad=0.04)
    ax2.set_title(f"2-D Heat Map of heat-flux predictions at Altitude {z}")
    
    diff = ax3.imshow(np.abs(true_z-pred_z), cmap=color, interpolation='nearest')
    fig.colorbar(diff, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    ax3.set_title(f"Abs difference between true and pred heat-flux")
    
    plt.tight_layout()
    plt.show()

#-------------CONCATENATE AND COARSE GRAIN-------------------------------

def concatenate_alt(dir,variable,t,i=0,sync=True):
    '''
    ## Description
    Concatenate all files in a directory over the altitude axis for a given instant
    File name format : '{variable}xy{alt}_T{t}.nc'.

    ## Parameters
    - dir (str): directory where the files are stored
    - variable (str) : name of the variable to concatenate
    - t (int) : time instant to concatenate
    '''
    files = [name for name in os.listdir(dir) if  name.__contains__(f"{variable}xy") and name.__contains__(f"_T{t}.nc")]
    lz = len(files)

    assert lz!=0, f"there is no file for this variable {variable} and time {t} instant"
    files.sort(key=lambda x: int(x.split(f"{variable}xy")[1].split(f"_T{t}.nc")[0]))

    #initialize array
    path_data = os.path.join(dir, files[0])
    nc_init = nc.Dataset(path_data)
    arr = nc_init[f'{variable}xy0'][:].filled()[:,:,:,:]       # modif xy0 if 0 is not first altitude layer
    # arr[0] = arr[0] - arr[0].mean(axis=(1,2)

    # concatenate
    for z in range(1,lz):
        path_data = os.path.join(dir, files[z])
        nc_init = nc.Dataset(path_data)
        arr2 = nc_init[f'{variable}xy{z+0}'][:].filled()[:,:,:,:]     # add number of first alt layer to z if 0 is not first altitude layer
        # arr2[0] = arr2[0] - arr2[0].mean(axis=(1,2)
        arr = np.concatenate((arr,arr2),axis=1)

    if sync :
        return arr
    else :
        return (i,arr)


def concatenate_time(dir,variable):
    '''
    ## Description
    Concatenate all files in a directory over the time axis for a given variable.
    File name format : '{variable}xy{altitude}_T{time}.nc'.

    ## Parameters
    - dir (str): directory where the files are stored
    - variable (str) : name of the variable to concatenate
    '''
    files = [name for name in os.listdir(dir) if  name.__contains__(f"{variable}xy")]
    temp = [int(re.split('(\d+)', name)[3]) for name in files]
    times = [t for n, t in enumerate(temp) if t not in temp[:n]]
    times.sort()

    tot_arr = concatenate_alt(0,dir,variable,times[0])

    for t in range(1,len(times)) :
        t_arr = concatenate_alt(0,dir,variable,times[t])
        tot_arr = np.concatenate((tot_arr,t_arr),axis=0)
        
    return tot_arr


def coarse_array(ds, L, i=0, sync=True):
    '''
    ## Description
    Coarsen a dataset over the altitude and time axis.
    The new shape is (time, altitude, y, x) where y and x are divided by L.

    ## Parameters
    - ds (np array) : dataset to coarsen
    - L (int) : coarsening factor
    '''
    #initialize coarse array
    coarse_ds = np.zeros((ds.shape[0], ds.shape[1], int(ds.shape[2]/L), int(ds.shape[3]/L)))
    for t in range(ds.shape[0]):
        for z in range(ds.shape[1]):
            for i in range(int(ds.shape[2]/L)):
                for j in range(int(ds.shape[3]/L)):
                    coarse_ds[t,z,i,j] = np.mean(ds[t,z,i*L:(i+1)*L,j*L:(j+1)*L])

    if sync :
        return coarse_ds
    else :
        return (i,coarse_ds)


def variable_samples(ds_coarse):
    '''
    ## Description
    Get the samples of a coarse_dataset over the altitude axis.
    The new shape is (time*y*x, z) where y and x are the horizontal dimensions divided by L.

    ## Parameters
    - ds (np array) : dataset to coarsen
    - L (int) : coarsening factor
    '''
    #initialize coarse array
    samples = np.zeros((ds_coarse.shape[0]*ds_coarse.shape[2]*ds_coarse.shape[3], ds_coarse.shape[1]))
    for t in range(ds_coarse.shape[0]):
        for i in range(ds_coarse.shape[2]):
            for j in range(ds_coarse.shape[3]):
                samples[t*ds_coarse.shape[2]*ds_coarse.shape[3] + i*ds_coarse.shape[3] + j] = ds_coarse[t,:,i,j]
    return samples


def input_dataset(datasets):
    '''
    ## Description
    Creates an input dataset for a nn from coarse datasets in the list variables.
    The new shape is (time*y*x, altitude*nbvar).

    ## Parameters
    - datasets (list) : list of coarse datasets
    '''
    l = len(datasets)
    assert l!=0, 'variables list is empty'
    #initialize dataset
    ds = variable_samples(datasets[0])
    for var in range(1,l):
        ds = np.concatenate((ds,variable_samples(datasets[var])), axis=1)
    return ds

#-------------WRITE FILES-------------------------------------------

def write_nc_file(data,coarsening_factor,var,time,nz=376):
    '''
    ## Description
    Write a netCDF file from a numpy array.

    ## Parameters
    - data (np.array) : array to write
    - coarsening_factor (int) : coarsening factor of the data
    - var (list of strings) : names of the variables
    - nz (int) : number of altitude layers
    '''
    #print (f'writing out time {time}')
    
    n_samples = data.shape[0]
    len_sample = len(var*nz)

    # open a netCDF file to write
    ncout = Dataset(f'data/L_{coarsening_factor}/input_ds_for_simple_nn_T{time}_L_{coarsening_factor}.nc', 'w', format='NETCDF4')

    # define axis size
    ncout.createDimension('index', n_samples)  
    ncout.createDimension('i', len_sample)  

    # create time axis
    index = ncout.createVariable('index', np.dtype('int16').char, ('index'))
    index.long_name = 'index'
    i = ncout.createVariable('i', np.dtype('int16').char, ('i'))
    i.long_name = 'i'

    # copy axis from original dataset
    index[:] = np.arange(0,n_samples)
    i[:] = np.arange(0,len_sample)
    
    # create variable array
    vout = ncout.createVariable('sample', np.dtype('double').char, ('index','i'))
    vout[:,:] = data
    
    ncout.close()

#-------------SPLIT TRAIN-TEST-----------------------------------------

def split_train_val_vieux(input_ds, batch_size):
    '''
    ## Description
    Split the dataset into training and validation sets.

    ## Parameters
    - input_ds (np array) : dataset to split
    - output_ds (np_array) : dataset to split
    '''
    # split the dataset
    n = len(input_ds)
    train, val = input_ds[:int((n//batch_size)*0.8)*batch_size], input_ds[int((n//batch_size)*0.8)*batch_size:]

    return train, val

def split_times(tmin,tmax,model_number):
    '''
    ## Description
    Split the times into training and test lists of times.

    ## Parameters
    - tmin (int) : first instant, usually 1
    - tmax (int) : last instant
    - model_number : number of the model trained with this split
    '''
    print(os.getcwd())
    times = [i for i in range(1,63)]
    perm = np.random.permutation(times)
    train_times = perm[:int(0.8*len(times))]
    test_times = perm[int(0.8*len(times)):]

    df_timetrain = pd.DataFrame([train_times]).T
    df_timetrain.columns = ['train']
    df_timetrain.to_csv(f'data/test_train_times/times_train_{model_number}.csv') 

    df_timetest = pd.DataFrame([test_times]).T
    df_timetest.columns = ['test']
    df_timetest.to_csv(f'data/test_train_times/times_test_{model_number}.csv')

def make_train_test_ds(coarse_factors, len_in, train_times, test_times, Directory):
    '''
    # Description
    Make the training and test datasets.

    # Parameters
    - coarse_factors (list) : list of coarsening factors
    - len_in (int) : length of the input dataset
    - train_times (list) : list of training times
    - test_times (list) : list of test times
    - Directory (string) : directory where to find the datasets (ex : data)
    '''
    #init train ds
    path_data = Directory+f'/L_{coarse_factors[0]}/input_ds_for_simple_nn_T{train_times[0]}_L_{coarse_factors[0]}.nc'
    nc_init = nc.Dataset(path_data)
    train_ds = nc_init['sample'][:].filled()

    for L in coarse_factors :
        for t in train_times :
            if L == coarse_factors[0] and t == train_times[0] :
                continue
            path_data = Directory+f'/L_{L}/input_ds_for_simple_nn_T{t}_L_{L}.nc'
            nc_init = nc.Dataset(path_data)
            time_ds = nc_init['sample'][:].filled()
            train_ds = np.concatenate((train_ds, time_ds), axis=0)

    #init test ds
    path_data = Directory+f'/L_{coarse_factors[0]}/input_ds_for_simple_nn_T{test_times[0]}_L_{coarse_factors[0]}.nc'
    nc_init = nc.Dataset(path_data)
    test_ds = nc_init['sample'][:].filled()

    for L in coarse_factors :
        for t in test_times:
            if L == coarse_factors[0] and t == test_times[0] :
                continue
            path_data = Directory+f'/L_{L}/input_ds_for_simple_nn_T{t}_L_{L}.nc'
            nc_init = nc.Dataset(path_data)
            time_ds = nc_init['sample'][:].filled()
            test_ds = np.concatenate((test_ds, time_ds), axis=0)

    # split train and test ds in input-output datasets
    input_train, output_train, input_val, output_val = train_ds[:,:len_in], train_ds[:,len_in:], test_ds[:,:len_in], test_ds[:,len_in:]
    input_train.shape

    # train : convert numpy array to torch tensor
    input = torch.from_numpy(input_train).float()
    output = torch.from_numpy(output_train).float()

    # test : convert numpy array to torch tensor
    input_test = torch.from_numpy(input_val).float()
    output_test = torch.from_numpy(output_val).float()

    return input, output, input_test, output_test


#-------------LAYER-WISE RELEVANT PROPAGATION-----------------------------------------

def rho(w,l):  return w #+ [None,0.1,0.0,0.0][l] * np.maximum(0,w)
def incr(z,l): return z #+ [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9

def lrp(model,input,z,one_alt=True):
    '''
    ## Description
    Layer-wise relevance propagation for a model taking the altitude as input and outputing the predicted heat flux at a single altitude.

    ## Parameters
    - model (torch.nn.Module) : model to propagate
    - input (np.array) : input to the model
    - output (np.array) : output of the model
    - z (int) : altitude
    '''
    if one_alt:
        X = torch.cat((input, torch.ones(1)*z), 0)
    else:
        X = input
    Y = model(X).detach().numpy()
    X = X.numpy()
    W = []
    B=[]

    for param_tensor in model.state_dict():
        if param_tensor.__contains__("weight"):
            W.append(model.state_dict()[param_tensor].cpu().detach().numpy())
        if param_tensor.__contains__("bias"):
            B.append(model.state_dict()[param_tensor].cpu().detach().numpy())
    L = len(W)    

    A = [X]+[None]*L
    for l in range(L):
        if l!=L-1:
            A[l+1] = np.maximum(0,A[l].dot(W[l].T)+B[l])
        else:
            A[l+1] = A[l].dot(W[l].T)+B[l]
        #print(A[l].max(),A[l].min())

    #print(A[L].max(),A[L].min())
    R = [None]*L + [A[L]*Y]
    #print(R[L].max(),R[L].min())

    for l in range(1,L)[::-1]:
        w = rho(W[l],l)
        b = rho(B[l],l)
        
        z = incr(A[l].dot(w.T)+b,l)   # step 1
        #z= z.reshape(z.shape[0])
        s = R[l+1] / z                # step 2
        c = s.dot(w)               # step 3
        R[l] = A[l]*c                # step 4
        #print(l,R[l].max(), R[l].min())

    #first layer : original inputs

    #print(A[0].max())
    w  = W[0]
    wp = np.maximum(0,w)
    wm = np.minimum(0,w)
    lb = A[0]*0-1
    hb = A[0]*0+1

    z = A[0].dot(w.T)-lb.dot(wp.T)-hb.dot(wm.T)+1e-9    # step 1
    #print(z.max(), z.min())
    s = R[1]/z                                          # step 2
    #print(s.max(), s.min())
    c,cp,cm  = s.dot(w),s.dot(wp),s.dot(wm)             # step 3
    #print(c.max(), c.min(), cp.max(), cp.min(), cm.max(), cm.min())
    R[0] = A[0]*c-lb*cp-hb*cm 
    #print(R[0].max())                          # step 4

    if one_alt:
        plot_ds = R[0][:-1].reshape(376,6)
    else:
        plot_ds = R[0].reshape(376,6)
    return plot_ds