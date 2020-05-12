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

seed = 42

print('strated at: ', datetime.datetime.now())

mainpath = 'data/'

filenames = []
for root, dirs, files in os.walk(mainpath):
    for file in files:
        if file.endswith(".h5"):
             filenames.append(os.path.join(root, file))
                
train, test = train_test_split(filenames, test_size = 0.1, random_state = seed)

with open("test_files.txt", 'w') as f:
    for item in test:
        f.write("%s\n" % item)

rf = RF(n_estimators=100, criterion='gini', max_depth=10, verbose=10, n_jobs=5, random_state=seed)

n = 401401
train_data = np.empty((n*len(train),4), float)
train_label = np.empty((n*len(train)), int)


counter = 0
for file in train:
    with h5py.File(file, 'r') as f:
        label = f['label'][()]
        data = f['data'][()]
        train_data[counter*n:(counter+1)*n] = data
        train_label[counter*n:(counter+1)*n] = label
        
    counter += 1
    print(str(int((counter/len(train))*100)) + '%', end='\r', flush = True)


#[544491999,   9633360,   3808992,    770355,    848288]
#[0.97308388, 0.01721617, 0.00680721, 0.00137673, 0.00151601]
train_label[np.where(train_label > 4)] = 4
index0 = np.where(train_label == 0)[0]
index1 = np.where(train_label == 1)[0]
index2 = np.where(train_label == 2)[0]
np.random.seed(seed)
index0 = np.random.choice(index0, size=int(0.005*len(index0)), replace = False)
np.random.seed(seed)
index1 = np.random.choice(index1, size=int(0.25*len(index1)), replace = False)
np.random.seed(seed)
index2 = np.random.choice(index2, size=int(0.5*len(index1)), replace = False)
indexn = np.where(train_label > 2)[0]
index = np.concatenate((index0,index1,index2,indexn), axis=None)
    
train_data = train_data[index]
train_label = train_label[index]
#unique, counts = np.unique(train_label, return_counts=True)
#[5444919, 4816680, 3808992,  770355,  848288]
#[0.34704811, 0.30700543, 0.24277744, 0.04910087, 0.05406816]

index3 = np.where(train_label == 3)[0]
index4 = np.where(train_label == 4)[0]
np.random.seed(seed)
index3 = np.concatenate((index3, np.random.choice(index3, size=int(1*len(index3)), replace = True)), axis=None)
np.random.seed(seed)
index4 = np.concatenate((index4, np.random.choice(index4, size=int(1*len(index4)), replace = True)), axis=None)
indexn = np.where(train_label < 3)[0]
index = np.concatenate((index3,index4,indexn), axis=None)

train_data = train_data[index]
train_label = train_label[index]

#[5444919, 4816680, 3808992, 1540710, 1696576]
#[0.31459196, 0.2782941 , 0.22007274, 0.08901785, 0.09802335]

print('prep complete, now fitting')
rf.fit(train_data, train_label)    
save = 'data/fitted_models/rf_model_down_upsample_2_set_depth_10.sav'
pickle.dump(rf, open(save, 'wb'))

print('finished at: ', datetime.datetime.now())