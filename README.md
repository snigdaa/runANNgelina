# runANNgelina
scripts to run the trained and tested RT emulator, ANNgelina

To run ANNgelina on your data, make a .npy file containing your input features, with each row containing the input features for a galaxy and each column corresponding to a feature.

Format for input features: [0: Mstar (in M_sol); 1: Mdust (in M_sol); 2: Z* (in ?); 3: Z* <10Myr (in ?); 4: SFR (in M_sol / yr)]

You can tack on any other information about your file after these indices if you desire, but you will not be able to access them from test_loader unless you code them into data.py as input features. For a robust understanding of how the NN is reading in data, please look through the code & comments in data.py. 

Do not modify any variables except f_features (and f_features_norm, if you would like to normalize your data to some other standard than the existing feature set. Refer to data.py - make_dataset() to understand how normalization is done)

To run the NN:

1) Load necessary modules/packages

2) in your terminal, run $ python run.py

The script should output plotted SEDs in ./PlottedSEDs, as well as csv files containing the estimated SED values from the NN