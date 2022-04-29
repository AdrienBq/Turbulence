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

os.chdir(Path(sys.path[0]).parent)


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
    ncout = Dataset(f'data/L_32/input_ds_for_simple_nn_T{time}_L_{coarsening_factor}.nc', 'w', format='NETCDF4')

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

def split_train_val(input_ds, batch_size):
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
