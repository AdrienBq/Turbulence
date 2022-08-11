# **Sub-grid parametrization for the convective Boundary Layer**

In this README you can find the research project that I lead in Pr. Gentine's lab at Columbia University in the Earth and Evironmental department.

<br />
<p align="center">
  <img src="1st_page.png"
    width="400" 
    height=auto>
</p>
 <br />

Our work here is based on LES simulations. The LES simulations will be used as the ground truth for training different Deep Learning models to parameterize sub-grid scale processes. Since LES simulations (which have a resolution of about 10m) can resolve clouds and convective processes in general, the goal is to use these simulations as ground truth to parameterize lower resolution simulations such as Global Circulation Models that have a resolution of about 10km. There is a huge gap between the resolution of Large-Eddy Simulations and Reynolds Averaged Navier-Stokes based simulations. Therefore, the idea is to bridge this gap by devising a model that can generalize across the different resolution scales. Based on a ground truth of a LES, we aim at a model that can predict accurate sub-grid fluxes on different larger scales.

The first part of the project consisted in getting familiar with the high resolution simulations available, preprocessing the data and designing a predictive model for sub-grid fluxes over and arbitrary resolution scale. The second part of the project was adapting this model so that it could adapt to different resolutions and be accurate on grids with a resolution of 100m as well as on grids with a resolution of 1km.

The reader can refer to the full report for a detailed walkthrough of the work.


## **1. Project Structure**

We provide the structure of our project in which you can find usable scripts but also exploration notebooks with comments.

```
.
├── README.md
├── full_report.pdf
├── explo : the exploration folder
    ├── write_nc_file.ipynb : notebook used to coarse-grain the high resolution data and store it
    ├── temperature_variations.ipynb : notebook which explores the variations of the mean temperature with the altitude
    ├── pca_dnn.ipynb :notebook which explores the principal compenants of the input data
    ├── lrp.ipynb : notebook which explores the most important variables in the predictions using Layer-wise Relevant Propagation
    ├── conv_net : notebook which explores a first convolutional network for predictions
    ├── eval_hori_velocity_divergence.ipynb : notebook which explores the link between velocity divergence in the horizontal plane and error of a predictive model
    ├── CNN_interpolation.py : python file used to design a convolutional model for vertical interpolation

├── modules
    ├── utils.py : file containing useful functions
    ├── model_train :folder containing the programs to run to train different networks
    ├── hyper_param_opt : folder containing programs to optimize the hyper-parameters of the models in the folder "model_train"
    ├── evaluation : 
        ├── eval_interpolation.py : file to evaluate the performance of different vertical interpolation methods
        ├── eval_models.py : file to evaluate the performance of different sub-grid flux prediction models
        
├── models : folder containing the main different models tested throught the internship that can be evaluated using the eval_models.py file

```

## **2. Requirements**
The project uses the following modules :
- pandas
- xarray
- netCDF4
- numpy
- matplotlib.pyplot
- tqdm
- os
- sys
- pathlib
- torch
- re
- multiprocessing


## **3 Usage**
### **3.1 Generate the data files**
Without the data files you cannot run the different models.
You need access to the high resolution simulations of Dr. Sara Shamekh in order to run the code.

- If you have access to the high-res data : 
run the write_nc_file.ipynb code to generate the coarse-grain data

- If you don't have access to the high-res data :
contact me at : burq.adrien@gmail.com
I will provide directly the coarse-grained data I used

You also need to use the split_times function in the utils.py file to create arrays of test and train times. They are used to define which data is used for training or testing.

### **3.2 Modify the code**
If you want to modify the hyper-parameters of a model, in general :
- in the main functions, you can change : 
    - the coarse-graining factors
    - the batch size
    - the number of epochs
- in the train functions :
    - the learning rate
    - the optimizer

To run the code, you just need to open the notebooks or run the .py files in a terminal.
