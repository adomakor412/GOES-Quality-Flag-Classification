#link to data repo https://gitlab.com/Andrii0/data/-/tree/master/fitted_models

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
                
train, test = train_test_split(filenames, test_size = 0.1, random_state = 42)

with open("test_files.txt", 'w') as f:
    for item in test:
        f.write("%s\n" % item)

rf = RF(n_estimators=100, criterion='gini', max_depth=10, verbose=10, n_jobs=5, random_state=42)

n = 401401
train_data = np.empty((n*len(train),4), float)
train_label = np.empty((n*len(train)), int)


counter = 0
for file in train:
    with h5py.File(file, 'r') as f:
        index = f['label'][()]
        data = f['data'][()]
        train_data[counter*n:(counter+1)*n] = data
        train_label[counter*n:(counter+1)*n] = index
        
    counter += 1
    print(str(int((counter/len(train))*100)) + '%', end='\r', flush = True)

rf.fit(train_data, train_label)    
save = 'data/fitted_models/rf_model_entire_set_depth_10.sav'
pickle.dump(rf, open(save, 'wb'))

print('finished at: ', datetime.datetime.now())