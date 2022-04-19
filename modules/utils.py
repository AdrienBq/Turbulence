import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import re

from tqdm.notebook import tqdm, trange
from netCDF4 import Dataset
import netCDF4 as nc4

import os
import sys
from pathlib import Path

os.chdir(Path(sys.path[0]).parent)




def print_one_alt(path_data,var,color):
    '''
    ## Description
    print first altitude layer of a data file

    ## Parameters
    - path_data (str) : path to the data file to be plotted
    - color (str) : colormap of the plot
    '''
    ds_init = xr.open_dataset(path_data)
    df_init = ds_init.to_dataframe()
    df_init.reset_index(inplace=True)
    couche0 = df_init[df_init['z']==0].drop(columns=['t','x','y','z'])
    arr = couche0.to_numpy()
    arr2 = arr.reshape(512, 512)
    plt.imshow(arr2 , cmap = color , interpolation = 'nearest' )
    plt.title(f"2-D Heat Map of {var}")
    plt.show()


def concatenate_alt(i,dir,variable,t,sync=True):
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

    assert lz!=0, "there is no file for this variable and time instant"
    files.sort()

    #initialize array
    path_data = os.path.join(dir, files[0])
    ds_init = xr.open_dataset(path_data)
    df_init = ds_init.to_dataframe()
    lengths = [len(df_init.index.levels[i]) for i in range(4)]
    arr = df_init.values.reshape(lengths[0],lengths[1],lengths[2],lengths[3])

    # concatenate
    for z in range(1,lz):
        path_data = os.path.join(dir, files[z])
        ds_init = xr.open_dataset(path_data)
        df_init = ds_init.to_dataframe()
        lengths = [len(df_init.index.levels[i]) for i in range(4)]
        arr2 = df_init.values.reshape(lengths[0],lengths[1],lengths[2],lengths[3])
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

    tot_arr = concatenate_alt(dir,variable,times[0])

    for t in range(1,len(times)) :
        t_arr = concatenate_alt(dir,variable,times[t])
        tot_arr = np.concatenate((tot_arr,t_arr),axis=0)
        
    return tot_arr

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
    print ('writing out')
    
    n_samples = data.shape[0]
    len_sample = len(var*nz)

    # open a netCDF file to write
    ncout = Dataset(f'data/input_ds_for_simple_nn_T{time}_L_{coarsening_factor}.nc', 'w', format='NETCDF4')

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


def write_coarse_file(data,coarsening_factor,var,nz=376):
    '''
    ## Description
    Write a netCDF file from a numpy array after coarsening.

    ## Parameters
    - data (np.array) : array to write
    - coarsening_factor (int) : coarsening factor of the data
    - var (str) : name of the variable
    - nz (int) : number of altitude layers
    '''
    print ('writing out')
    
    T = data.shape[0]
    nz = data.shape[1]
    ny = data.shape[2]
    nx = data.shape[3]

    # open a netCDF file to write
    ncout = Dataset('data/'+var+'_L_'+coarsening_factor+'.nc', 'w', format='NETCDF4')

    # define axis size
    ncout.createDimension('time', T)  
    ncout.createDimension('x', nx)
    ncout.createDimension('y', ny)
    ncout.createDimension('z', nz)

    # create time axis
    time = ncout.createVariable('time', np.dtype('int16').char, ('time',))
    time.long_name = 'time'
    time.units = 'sec'
    # time.calendar = 'standard'
    time.axis = 'T'

    # create latitude axis
    x = ncout.createVariable('x', np.dtype('int16').char, ('x'))
    x.standard_name = 'x'
    x.long_name = 'x axis'
    x.units = 'meter'
    x.axis = 'X'

    # create longitude axis
    y = ncout.createVariable('y', np.dtype('int16').char, ('y'))
    y.standard_name = 'y'
    y.long_name = 'y axis'
    y.units = 'meters'
    y.axis = 'Y'

    # create z axis
    z = ncout.createVariable('z', np.dtype('int16').char, ('z'))
    z.standard_name = 'z'
    z.long_name = 'z axis'
    z.units = 'level'
    z.axis = 'z'

    # copy axis from original dataset
    time[:] = np.arange(0,T)
    z[:] = np.arange(0,nz)
    y[:] = np.arange(0,ny)
    x[:] = np.arange(0,nx)
    
    # create variable array 
    vout = ncout.createVariable(var, np.dtype('double').char, ('time','z', 'y', 'x'))
    vout.long_name = var
    vout.units = var
    vout[:,:,:,:] = data

    ncout.close()

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


def coarse_array(ds, L):
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
    return coarse_ds

def coarse_array2(i,ds, L):
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

# not tested yet
def reconstruct_arrays(arr, n_features, L):
    '''
    ## Description
    Reconstruct a dataset over the altitude and time axis. 
    Need to be used on arrays of shape (time*y*x, z*n_features) where y and x are the horizontal dimensions after coarsening by the factor L.
    The new shape is (time, altitude, y, x) where y and x are the horizontal dimensions divided by L.

    ## Parameters
    - arr (np array) : dataset to reconstruct. The shape is (time*y*x, z*n_features)
    - feature (str) : name of the variable to reconstruct
    - n_features (int) : number of features in the dataset

    ## Output
    Arrays of size (time, altitude, y, x) for each feature
    '''
    #features are typically ['u', 'v', 'w', 'theta']
    arr_reconstructed = np.zeros((n_features, arr.shape[0]/(512/L)**2, arr.shape[2]/n_features, int(512/L), int(512/L)))
    for f in range(n_features):
        for t in range(int(arr.shape[0]/(512/L)**2)):
            for z in range(arr.shape[2]/n_features):
                for i in range(int(512/L)):
                    for j in range(int(512/L)):
                        arr_reconstructed[f,t,z,i,j] = arr[t*int(512/L)**2+i*int(512/L)+j,f*arr.shape[2]/n_features+z]
    return arr_reconstructed


def input_dataset(datasets):
    '''
    ## Description
    Creates an input dataset for a nn from coarse datasets in the list variables.
    The new shape is (time*y*x, altitude*nbvar).

    ## Parameters
    - dir (str) : directory where the data is stored
    - variables (list) : list of variables to include in the dataset
    - L (int) : coarsening factor
    '''
    l = len(datasets)
    assert l!=0, 'variables list is empty'
    #initialize dataset
    ds = variable_samples(datasets[0])
    for var in range(1,l):
        ds = np.concatenate((ds,variable_samples(datasets[var])), axis=1)
    return ds