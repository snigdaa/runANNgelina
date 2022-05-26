import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time, h5py

# This class creates the dataset 
class make_dataset():
    #mode: train, test, validate, or all -- test for running final model
    #seed: random seed to split training data; set explicitly to reproduce split, otherwise use randint
    #f_input: features
    #f_input_norm: if multiple datasets are used, use f_input_norm to normalize; should have same shape & format as f_input
    #f_output: labels (SEDs)
    def __init__(self, mode, seed, f_input, f_input_norm, f_output):
        # read the value of the global properties; normalize them
        features  = np.load(f_input)
        features[:,[0,1,2,3,4]] = np.log10(features[:,[0,1,2,3,4]])
        print(features.shape)

        if f_input_norm is None:
            features = features[:,[0,1,2,3,4]]
            mean, std = np.mean(features, axis=0), np.std(features, axis=0)
        else:
            featurenorm = np.load(f_input_norm)
            featurenorm[:, [0,1,2,3,4]] = np.log10(featurenorm[:, [0,1,2,3,4]])
            featurenorm = featurenorm[:, [0,1,2,3,4]]
            mean,std = np.mean(featurenorm, axis=0), np.std(featurenorm, axis=0)
        features = (features - mean)/std
       
        # read SED data, scale it (will not affect test, validate, all. Placeholder for create_dataset()
        SED = np.loadtxt(f_output)
        SED = np.log10(SED)

        # get the size and offset depending on the type of dataset
        sims = features.shape[0]
        if   mode=='train':  size, offset = int(sims*0.70), int(sims*0.00)
        elif mode=='valid':  size, offset = int(sims*0.15), int(sims*0.70)
        elif mode=='test':   size, offset = int(sims*0.15), int(sims*0.85)
        elif mode=='all':    size, offset = int(sims*1.00), int(sims*0.00)
        else:                raise Exception('Wrong name!')

        # randomly shuffle the sims. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(sims) #only shuffle realizations, not rotations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # select the data in the considered mode
        features = features[indexes]
        SED      = SED[indexes]

        # define size, input and output matrices
        self.size   = size
        self.input  = torch.tensor(features, dtype=torch.float)
        self.output = torch.tensor(SED,      dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
# mode ---------------> 'train', 'valid', 'test' or 'all'
# seed ---------------> random seed to split data among training, validation and testing
# f_features ---------------> file containing the input features - global properties
# f_features_norm ----------> to be a file to normalize features with both tng and fire are in, None for now
# f_params -----------> files with the skirt seds, or true outputs 
# batch_size ---------> batch size
# shuffle ------------> whether to shuffle the data or not (highly recommended, should only be false for testing purposes)
# workers --------> number of CPUs to load the data in parallel (to use when running on gpu (= num cpus/gpu), otherwise set to 1)
def create_dataset(mode, seed, f_features, f_features_norm, f_labels, batch_size, shuffle, workers=1):
    data_set = make_dataset(mode, seed, f_features, f_features_norm, f_labels)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers)


#Calculate fractional difference of obs and exp & fractional difference of std of exp
def calcFDiff(pred, exp, logged=True):
    totfracdiff = []
    stdfracdiff = []

#if you're not converting your inputs to numpy before plugging into the function, then uncomment this
#     try:
#         exp = exp.numpy()
#     except:
#         exp = exp.cpu().numpy()
        
    loggedstdExp = float((exp).std())
    unloggedstdExp = float(np.log10(exp.std()))
    
    for q, val in enumerate(pred):
        if logged:
            fdiff = float(val - exp[q])
        else:
            fdiff = float(np.log10(val) - np.log10(exp[q]))
        
        try:
            fdiff = np.asarray(fdiff)
        except Exception as e:
            print(e)
        
        # given an expected and true SED, outputs the fractional differences per wavelength in an array
        totfracdiff.append(fdiff)

    
    toreturn = {'fracdiff': np.asarray(totfracdiff)}
    
    return toreturn