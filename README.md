# runANNgelina
This repository contains scripts to run the trained and tested radiative transfer emulator, ANNgelina. Currently is optimized to be used on IllustrisTNG galaxies.

### 1) make a .npy file containing your input features, with each row containing the input features for a galaxy and each column corresponding to a feature.

Go ahead and delete data/tngFeatures.npy. KEEP tngFluxesOriginal.txt to maintain tensor shape when evaluating the model. It will not be referenced in any mode other than 'train.'

NOTE - Format for input features, keep the same ordering: 
> [**0: Mstar** (solar mass); **1: Mdust** (solar mass); **2: Z*** (metal mass / solar mass); **3: Z*** **<10Myr** (metal mass / solar mass); **4: SFR** (solar mass / yr)] 

You can tack on any other information about your file after these indices if you desire, but you will not be able to access them from test_loader unless you code them into data.py as input features. For a robust understanding of how the NN is reading in data, please look through the code & comments in data.py. 

### 2) Open run.py. ONLY modify f_features to point to your input features file
And add pointer for f_features_norm, if you would like to normalize your data with some other standard deviation/mean of features than the input set. Refer to Methods in paper or function make_dataset() in data.py to understand in detail how normalization is done. 

### 3) Open terminal

### 4) Load necessary modules/packages

### 5) In terminal: $ python run.py

The script should output plotted SEDs in ./PlottedSEDs, as well as csv files containing the estimated SED values from the NN

Note: Pull requests for modification are welcome. This code is being actively modified and updated, so if you run into bugs or issues please be patient or contribute to the solution!

To read the methodology and analysis of ANNgelina, please refer to Sethuram, Cochrane, et al. (in prep)
