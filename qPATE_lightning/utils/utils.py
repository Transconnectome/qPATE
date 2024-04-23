import torch
import numpy as np
import torchvision
import time
import os
import copy
import math
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

#################################
##### Saving and Plotting   #####
#################################

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig("cats_vs_dogs_data_demo" + ".pdf", format='pdf')
    plt.pause(5)  # pause a bit so that plots are updated
    

   
    
def load_pickled_results(filename):
    '''
        load Sam's results using pickle
        from the DP_Results folder
        Return is a n_epochs length vector
    '''
    with open(filename, "rb") as fp:
        results = pickle.load(fp)
    print("Results...")
    print(results)
    return results
    
def save_data(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)
        
def load_data(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

