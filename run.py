#run neural network on your data

import numpy as np
import sys, os, time, h5py
from pathlib import Path
import torch
import torch.nn as nn
import data, architecture
import optuna
import matplotlib.pyplot as plt

################################### INPUT ############################################
# data parameters - change as needed

# input features you need: [0: Mstar (in M_sol); 1: Mdust (in M_sol); 2: Z* (in ?); 3: Z* <10Myr (in ?); 4: SFR (in M_sol / yr)]
# specific wavelength array for output is available in ./data/wave_tng.dat

f_features      = './data/tngFeatures.npy'          # path to your input features 
f_features_norm = None                              # your normalization file, if you have one. Otherwise will normalize w/ feature file. Check data.py - make_dataset() to understand how this works

#architecture parameters - do not touch
mode        = 'all'                                 # 'train','valid','test' or 'all' -- check data.py for what this means
input_size  = 5                                     # dimensions of input data (number of features)
numHL       = 1                                     # number of hidden layers in NN
hidden      = [954]                                 # # of nodes in hl
dr          = [0.20103]                             # dropout rate
batch_size  = 256                                   # batch size for training model
epochs      = 1500                                  # num epochs for training model
seed        = 2                                     # seed to shuffle data in train/valid/test; set to any value, remember it if you want to reproduce the shuffle. Will not affect 'all' but has to be included as a parameter for create_dataset anyways
f_labels    = './data/tngFluxesOriginal.txt'        # existing SEDs - placeholder, do not modify

# optuna & file parameters - do not touch
fname       = 'tng_dynamicmodel2'                   # model name
output_size = 200                                   # size of wavelength array

wavelengths_file = './data/wave_tng.dat'            # file containing wavelength array to load into np array  
######################################################################################

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# get neural network containing state_dict
fmodel = 'model/{}.pt'.format(fname)          

# generate the architecture
model = architecture.dynamic_model2(input_size, output_size, numHL, hidden, dr)
model.to(device)    

# load fmodel
if os.path.exists(fmodel):  
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
else:
    raise Exception('Specified model does not exist: {}'.format(fmodel))

# define loss function
criterion = nn.MSELoss() 

# get the data
test_loader = data.create_dataset(mode, seed, f_features, f_features_norm, f_labels, batch_size, shuffle=False)

# tell the model you are only using it, not training it
model.eval()

# these are global properties I wanted to print on my test plots, so I'm tracking them
ogMS    = []
ogMD    = []
ogSFR   = []

# get wavelengths into logged and unlogged arrays
wavelength_unlogged = np.genfromtxt(wavelengths_file, dtype=None,delimiter=' ')
wavelength_logged   = np.log10(np.array(wavelength_unlogged))

# create directory to store plots
try:
    Path('./PlottedSEDs').mkdir(parents=False, exist_ok=False)
except Exception as e:
    print(e)


# get normalization factors for features file, since that is what's being normalized in data_loader here
if f_features_norm is None:
    features  = np.load(f_features)
else:
    features  = np.load(f_features_norm)
logfeat = np.log10(features[:, [0,1,2,3,4]])
meanms = np.mean(logfeat[:,0])
meanmd = np.mean(logfeat[:,1])
meansfr = np.mean(logfeat[:,4])
stdms = np.std(logfeat[:,0])
stdmd = np.std(logfeat[:,1])
stdsfr = np.std(logfeat[:,4])

# for plot naming scheme
counter = -1

# to store calculated SEDs
estimatedLoggedSEDs = []
estimatedUnloggedSEDs = []
               
with torch.no_grad():
    # open data; if larger than batch size, will iterate over x, y multiple times hence the counter
    for x, y in test_loader:
        bs      = x.shape[0]    # batch size
        x       = x.to(device)  # input features
        y_NN    = model(x)      # NN estimate
        
        # iterate over each galaxy in batch
        for i in range(0, len(x)):
            counter += 1

            # denormalize features
            try:
                sfr = x[i][4].numpy()
                ms = x[i][0].numpy()
                md = x[i][1].numpy()
            except:
                sfr = x[i][4].cpu().numpy()
                ms = x[i][0].cpu().numpy()
                md = x[i][1].cpu().numpy()

            sfr1 = sfr*stdsfr + meansfr
            denormed_sfr = 10**sfr1
            ms1 = ms*stdms + meanms
            denormed_ms = 10**ms1
            md1 = md*stdmd + meanmd
            denormed_md = 10**md1
            denormed_sfr = round(denormed_sfr, ndigits = 2)
            denormed_ms = np.format_float_scientific(denormed_ms, precision=2)
            denormed_md = np.format_float_scientific(denormed_md, precision=2)
            
            # set up plot
            fig, axs = plt.subplots(2, figsize=(8,10))
            
            # get model output into np array
            try:
                nnestimate = model(x[i]).numpy()

            except:
                nnestimate = model(x[i]).cpu().numpy()
            
            unloggedestimate = [10**each for each in nnestimate]
            
            estimatedLoggedSEDs.append(nnestimate)
            estimatedUnloggedSEDs.append(unloggedestimate)

            # plot logged & unlogged SEDs
            axs[0].plot(wavelength_logged,nnestimate,'r-')
            props = dict(boxstyle='round', facecolor = 'mistyrose', alpha=0.4)

            axs[0].text(wavelength_logged.min()+0.01, nnestimate.min()-0.2, \
                'SFR: %s $M_{\odot} yr^{-1}$\n$M_{dust}$: %s $M_{\odot}$\n$M_{star}$: %s $M_{\odot}$' % (denormed_sfr, denormed_md, denormed_ms), fontsize=11,bbox=props)
            axs[0].set_ylim([nnestimate.min()-1,nnestimate.max()+1])
            axs[0].set_xlabel(r'$\lambda_{\rm{rest}}/log(\mu m)$',size=18)
            axs[0].set_ylabel(r'$\lambda*F_{\lambda}/log(Wm^{-2})$',size=18)
            
            axs[1].plot(wavelength_logged, unloggedestimate, 'g-')
            axs[1].set_xlabel(r'$\lambda_{\rm{rest}}/log(\mu m)$',size=18)
            axs[1].set_ylabel(r'$\lambda*F_{\lambda}/ [ Wm^{-2}$ ]',size=18)
            
            axs[0].set_title('Logged NN Estimate', size=20)
            axs[1].set_title('Unlogged NN Estimate', size=20)
            
            plt.gcf()
            plt.show()
            
            fig.savefig("./PlottedSEDs/Galaxy_{}.png".format(counter))
            fig.clf()
        print ('shape of each output')
        print (np.shape(x))
        print (np.shape(y_NN))

# save SEDs to csv
np.savetxt('estimatedLoggedSEDs.csv', estimatedLoggedSEDs, delimiter = ',')
np.savetxt('estimatedUnlogedSEDs.csv', estimatedUnloggedSEDs, delimiter = ',')
