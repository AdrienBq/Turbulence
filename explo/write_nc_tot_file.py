import pandas as df
import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm, trange

import os
import sys
from pathlib import Path

os.chdir(Path(sys.path[0]).parent)
import modules.utils as utils


Directory = "/glade/scratch/sshamekh/LES_512_ug16wtspt01_data"
L = 32

u_ds = utils.concatenate_time(Directory, 'u')
v_ds = utils.concatenate_time(Directory, 'v')
w_ds = utils.concatenate_time(Directory, 'w')
theta_ds = utils.concatenate_time(Directory, 'theta')
assert u_ds.shape == v_ds.shape == w_ds.shape == theta_ds.shape, 'u,v,w,theta have different shape'

u_coarse = utils.coarse_array(u_ds, L)
v_coarse = utils.coarse_array(v_ds, L)
w_coarse = utils.coarse_array(w_ds, L)

wtheta_ds = w_ds*theta_ds
tke_ds = utils.coarse_array(u_ds*u_ds, L) - u_coarse*u_coarse + utils.coarse_array(v_ds*v_ds, L) - v_coarse*v_coarse + utils.coarse_array(w_ds*w_ds, L) - w_coarse*w_coarse
tke_in = utils.variable_samples(tke_ds, 1)
output_ds = utils.variable_samples(wtheta_ds, L)

variables = ['u', 'v', 'w', 'theta']  # add 's' 
input_ds = utils.input_dataset(Directory, variables, L)
tot_ds = np.concatenate((np.concatenate((input_ds,tke_in), axis=1), output_ds), axis=1)

variables.append('tke')
variables.append('wtheta')
print(variables)

utils.write_nc_file(tot_ds,str(L),variables,4)