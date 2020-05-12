import os
import glob
import sys
import random
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
import pickle
import datetime 

print('strated at: ', datetime.datetime.now())

mainpath = 'data/'

filenames = []
for root, dirs, files in os.walk(mainpath):
    for file in files:
        if file.endswith(".h5"):
             filenames.append(os.path.join(root, file))

n = 401401
data = np.empty((n*len(filenames),4), float)
label = np.empty((n*len(filenames)), int)


counter = 0
for file in filenames:
    with h5py.File(file, 'r') as f:
        index = f['label'][()]
        data = f['data'][()]
        label[counter*n:(counter+1)*n] = index
        
    counter += 1
    print(str(int((counter/len(filenames))*100)) + '%', end='\r', flush = True)

unique, counts = np.unique(label, return_counts=True)
print(unqiue)
print(counts)