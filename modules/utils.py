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


def plot_baseline(Directory, test_times, len_out, z, t, L, mean_out, std_out, color='RdBu_r'):
    '''
    ## Description
    Plot the prediction and true dataset for a given altitude and returns the corresponding array

    ## Parameters
    - Directory (str) : directory where the files are stored
    - test_times (list) : list of time instants in the test set
    - len_out (int) : length of the output vector
    - z (int) : altitude index
    - L (int) : coarsening factor
    - z (int) : altitude index
    - mean_out (int) : mean of the output dataset
    - std_out (int) : standard deviation of the output dataset
    - color (str) : color map
    '''
    largeur = int(512/L)
    path_data = Directory+f'/L_{L}/input_ds_for_simple_nn_T{test_times[t]}_L_{L}.nc'   #'data/L_32_new/input_ds_for_simple_nn_T10_L_32.nc'
    nc_init = nc.Dataset(path_data)
    true_heat_flux = nc_init[f'sample'][:].filled()[:,-len_out:][:,z].reshape(largeur,largeur)        # -len_out because : last len_out values are wtheta, z is wtheta at alt z
    true_heat_flux -= mean_out
    true_heat_flux /= std_out

    w_arr = nc_init[f'sample'][:].filled()[:,2*len_out:3*len_out][:,z].reshape(largeur,largeur)        # variables are [u,v,w,theta,s,tka,wtheta] so 2*len_out is w
    theta_arr = nc_init[f'sample'][:].filled()[:,3*len_out:4*len_out][:,z].reshape(largeur,largeur)    # variables are [u,v,w,theta,s,tka,wtheta] so 3*len_out is theta
    baseline_heat_flux = w_arr*theta_arr
    baseline_heat_flux -= mean_out
    baseline_heat_flux /= std_out

    print('Mean,min,max temperature fluctuation :',theta_arr.mean(),theta_arr.min(),theta_arr.max())
    print('Mean,min,max true heat flux :',true_heat_flux.mean(),true_heat_flux.min(),true_heat_flux.max())
    print('Mean,min,max baseline heat flux :',baseline_heat_flux.mean(),baseline_heat_flux.min(),baseline_heat_flux.max())

    fig,axes = plt.subplots(1,3,figsize=(16,4))

    im0 = axes[0].imshow(true_heat_flux , cmap = color , interpolation = 'nearest' )
    fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    axes[0].set_title(f"2-D Heat Map of true w*theta at t={test_times[t]} and z={z}")

    im1 = axes[1].imshow(baseline_heat_flux , cmap = color , interpolation = 'nearest' )
    fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    axes[1].set_title(f"2-D Heat Map of baseline w*theta at t={test_times[t]} and z={z}")

    im2 = axes[2].imshow(np.abs(true_heat_flux - baseline_heat_flux) , cmap = color , interpolation = 'nearest' )
    fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    axes[2].set_title(f"Diff between objective and baseline w*theta at t={test_times[t]} and z={z}")

    fig.tight_layout()
    plt.show()
    plt.savefig('explo/images/baseline_heat_flux.png')

    return baseline_heat_flux


def plot_output(pred_ds,true_ds,L,z,fig_name,color='RdBu'):
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
    
    diff = ax3.imshow(np.abs(true_z-pred_z), cmap=color, interpolation='nearest', vmin=0, vmax=3.5)
    fig.colorbar(diff, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    ax3.set_title(f"Abs difference between true and pred heat-flux")
    
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_name)


def plot_loss_div(input_ds,true_ds,model,L,fig_name):
    '''
    # Description
    Plot the pixel-wise prediction error versus the horizontal velocity divergence around said pixel.

    # Parameters
    - input_ds (np array) : prediction dataset
    - true_ds (np array) : true value dataset
    - model (torch.nn) : model used for prediction
    - L (int) : coarsening factor
    - fig_name (str) : name of the figure to save
    '''
    exp_shapex = int(512/L)
    exp_shapey = int(512/L)
    exp_shapez = 376

    pred_ds = model(input_ds)

    u_ds = input_ds[:,0,:]
    v_ds = input_ds[:,1,:]

    nb_im = pred_ds.shape[0]//(exp_shapex**2)

    losses = []
    divs = []

    for i in range(nb_im):
        for z in range(exp_shapez):
            pred_image = pred_ds[i*exp_shapex**2:(i+1)*exp_shapex**2,z].cpu().detach().numpy().reshape((exp_shapex,exp_shapey))
            true_image = true_ds[i*exp_shapex**2:(i+1)*exp_shapex**2,z].cpu().detach().numpy().reshape((exp_shapex,exp_shapey))
            u_image = u_ds[i*exp_shapex**2:(i+1)*exp_shapex**2,z].cpu().detach().numpy().reshape((exp_shapex,exp_shapey))
            v_image = v_ds[i*exp_shapex**2:(i+1)*exp_shapex**2,z].cpu().detach().numpy().reshape((exp_shapex,exp_shapey))

            for x in range(1,exp_shapex-1):
                for y in range(1,exp_shapey-1):
                    losses.append((pred_image[x,y]-true_image[x,y])**2)
                    u_ker = u_image[x-1:x+1,y-1:y+1]
                    v_ker = v_image[x-1:x+1,y-1:y+1]
                    u_sig = np.sqrt(np.var(u_ker))
                    v_sig = np.sqrt(np.var(v_ker))
                    u_mean = np.abs(np.mean(u_ker))
                    v_mean = np.abs(np.mean(v_ker))
                    divs.append(u_sig + v_sig)
    print(max(divs),min(divs))

    nb_bins = 50
    div_bins = [i*max(divs)/nb_bins for i in range(nb_bins+1)]
    loss_bins = [[] for _ in range(nb_bins+1)]
    for i in range(len(losses)):
        for j in range(nb_bins):
            if divs[i] >= div_bins[j] and divs[i] < div_bins[j+1]:
                loss_bins[j].append(losses[i])
    div_plot = [(i+1/2)*max(divs)/nb_bins for i in range(nb_bins)]
    loss_plot = [np.mean(loss_bins[i]) for i in range(nb_bins)]
    loss_plot_err = [2*np.std(loss_bins[i])/np.sqrt(len(loss_bins[i])) for i in range(nb_bins)]
    print(loss_plot_err)
    
    plt.scatter(div_plot,loss_plot, vmin=0, vmax=0.4)
    plt.errorbar(div_plot,loss_plot, yerr=loss_plot_err, fmt="o")
    plt.title('losses vs horizontal speed divergence')
    plt.xlabel('horizontal speed divergence')
    plt.ylabel('loss')
    plt.savefig(fig_name)
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

def lrp(model,input,sample=200,z=0,one_alt=False):
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
    if X.shape[0] != 376:
        X = X[sample]
        Y = Y[sample]
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

    if R[0].shape[0] != 376:
        if one_alt:
            plot_ds = R[0][:-1].reshape(376,6)
        else:
            plot_ds = R[0].reshape(376,6)
    else:
        plot_ds = R[0]
    return plot_ds



#-----------------------SUPER-RESOLUTION VERTICALE----------------------------------------------------

def interpolation_knn(input,N_output,max_in_height=1,max_out_height=1,n_neighboors=5):
    '''
    # Description
    Interpolation of the input data using the K nearest neighboors.

    # Parameters
    - input (np.array) : input data
    - N_output (int) : number of output points  
    - max_height (int) : maximum height of the input and output data
    - n_neighboors (int) : number of neighboors to use for the interpolation
    '''
    eps = 1e-3
    N_input = input.shape[0]
    z_inputs = [i*max_in_height/(N_input-1) for i in range(N_input)]
    out = np.zeros(N_output)
    for i in range(N_output):
        z=i*max_out_height/(N_output-1)
        distances = {j:np.abs(z-z_inputs[j]) for j in range(N_input)}
        distances = sorted(distances.items(), key=lambda x: x[1])
        nearest = [distances[j] for j in range(n_neighboors)]
        out[i] = np.sum(input[int(nearest[j][0])]/(nearest[j][1]+eps) for j in range(n_neighboors))/np.sum(1/(nearest[j][1]+eps) for j in range(n_neighboors))
    return out

def interpolation_linear(input,N_output,max_in_height=1,max_out_height=1):
    '''
    # Description
    Interpolation of the input data using a linear interpolation.

    # Parameters
    - input (np.array) : input data
    - N_output (int) : number of output points
    - max_height (int) : maximum height of the input and output data
    '''
    N_input = input.shape[0]
    z_inputs = [i*max_in_height/(N_input-1) for i in range(N_input)]
    out = torch.zeros(N_output)
    for i in range(N_output):
        if i==0:
            out[i] = input[0]
        else:
            z=i*max_out_height/(N_output-1)
            j = 0
            while z_inputs[j]<z:
                j+=1
                if j==N_input:
                    break
            if j==N_input:
                out[i] = input[-1]
            else:
                out[i] = input[j-1]+(z-z_inputs[j-1])*(input[j]-input[j-1])/(z_inputs[j]-z_inputs[j-1])
    return out

def interpolation_cubic(input,N_output,max_in_height=1,max_out_height=1):
    '''
    # Description
    Interpolation of the input data using a cubic interpolation.

    # Parameters
    - input (np.array) : input data
    - N_output (int) : number of output points
    - max_height (int) : maximum height of the input and output data
    '''
    N_input = input.shape[0]
    z_inputs = [i*max_in_height/(N_input-1) for i in range(N_input)]
    out = np.zeros(N_output)
    C_func=[]

    for i in range(1,N_input):
        A = np.array([[1,z_inputs[i-1],z_inputs[i-1]**2,z_inputs[i-1]**3],
                        [1,z_inputs[i],z_inputs[i]**2,z_inputs[i]**3],
                        [0,1,2*z_inputs[i-1],3*z_inputs[i-1]**2],
                        [0,1,2*z_inputs[i],3*z_inputs[i]**2]])
        if i==1:
            B = np.array([input[i-1],input[i],(input[i]-input[i-1])/(z_inputs[i]-z_inputs[i-1]),(input[i]-input[i-1])/(z_inputs[i+1]-z_inputs[i-1])])
        elif i==N_input-1:
            B = np.array([input[i-1],input[i],(input[i]-input[i-1])/(z_inputs[i]-z_inputs[i-1]),(input[i]-input[i-1])/(z_inputs[i]-z_inputs[i-2])])
        else:
            B = np.array([input[i-1],input[i],(input[i]-input[i-1])/(z_inputs[i]-z_inputs[i-1]),(input[i]-input[i-1])/(z_inputs[i+1]-z_inputs[i-1])])
        X = np.linalg.solve(A,B)
        C_func.append(X)

    for i in range(N_output):
        z=i*max_out_height/(N_output-1)
        j = 0
        while z_inputs[j]<z:
            j+=1
            if j==N_input:
                break
        if j==N_input:
            out[i] = input[-1]
        elif j==0:
            out[i] = input[0]
        else :
            out[i] = C_func[j-1][0]+C_func[j-1][1]*z+C_func[j-1][2]*z**2+C_func[j-1][3]*z**3
    return out


def interpolation_CNN(input,N_output,max_in_height,max_out_height,model):
    '''
    # Description
    Interpolation of the input data using a CNN.

    # Parameters
    - input (np.array) : input data
    - N_output (int) : number of output points
    - max_height (int) : maximum height of the input and output data
    - model (torch.nn) : CNN model
    '''
    N_input = input.shape[0]
    '''if N_input==64:
        model = ...
    elif N_input==128:
        model = ...
    elif N_input==256:
        model = ...
    
    out = model(input)'''
    return 

def interpolation_GNN(input,N_output,max_in_height,max_out_height):
    '''
    # Description
    Interpolation of the input data using a GNN.

    # Parameters
    - input (np.array) : input data
    - N_output (int) : number of output points
    - max_height (int) : maximum height of the input and output data
    - model (torch.nn) : GNN model
    '''
    N_input = input.shape[0]
    '''if N_input==64:
        model = ...
    elif N_input==128:
        model = ...
    elif N_input==256:
        model = ...
    
    out = model(input)'''
    return 


def interpolation(input,method,max_in_height=1,max_out_height=1,model=None,No=376,n_neighboors=5):
    '''
    # Description
    Interpolation of the input dataset to map the input to a higher resolution of No points.
    Needs the lowest point to be at 0m.
    We assume that the points in the input data are equidistant vertically.
    CNN and GNN methods work only for inputs of length 256, 128 or 64. Therefore if the input length is different, we interpolate on the closest length supported.

    # Parameters
    - input (np.array) : vertical vector of unknown size to map on a 376 points vertical vector
    - method (str) : method of interpolation : 'knn' ,'linear', 'cubic', 'CNN' or 'GNN'
    - No (int) : number of points to map the input on, default is 376
    '''
    if max_in_height > max_out_height:
        nb_in = input.shape[0]
        frac_to_keep = max_out_height/max_in_height
        input = input[:np.ceil(nb_in*frac_to_keep)]
        max_in_height = np.ceil(max_in_height*frac_to_keep)
    
    if method == 'knn':
        return interpolation_knn(input,No,max_in_height,max_out_height,n_neighboors)
    elif method == 'linear':
        return interpolation_linear(input,No,max_in_height,max_out_height)
    elif method == 'cubic':
        return interpolation_cubic(input,No,max_in_height,max_out_height)
    elif method == 'CNN':
        return interpolation_CNN(input,No,max_in_height,max_out_height,model)
    elif method == 'GNN':
        print('Method not yet implemented')
        #return interpolation_GNN(input,No,max_in_height,max_out_height,model)
        return None
    else:
        print('Method not implemented')
        return None