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

var = 'u'

ds = utils.concatenate_time(Directory, var)

coarse_ds = utils.coarse_array(ds, L)

utils.write_coarse_file(coarse_ds,str(L),var,4)