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
start = datetime.datetime.now()
print('strated at: ', start)

mainpath = 'data_window/'

filenames = []
for root, dirs, files in os.walk(mainpath):
    for file in files:
        if file.endswith(".h5"):
             filenames.append(os.path.join(root, file))
                
train, test = train_test_split(filenames, test_size = 0.1, random_state = seed)

with open("test_files_2.txt", 'w') as f:
    for item in test:
        f.write("%s\n" % item)

rf = RF(n_estimators=100, criterion='gini', max_depth=10, verbose=10, n_jobs=5, random_state=seed)

n = 399*999
train_data = np.empty((n*len(train),36), float)
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

#[147840980,   1861759,    844554,    211337,    311149]
#[0.9786271 , 0.01232383, 0.00559049, 0.00139894, 0.00205964]

train_label[np.where(train_label > 4)] = 4
index0 = np.where(train_label == 0)[0]
index1 = np.where(train_label == 1)[0]
index2 = np.where(train_label == 2)[0]
np.random.seed(seed)
index0 = np.random.choice(index0, size=int(0.005*len(index0)), replace = False)
np.random.seed(seed)
index1 = np.random.choice(index1, size=int(0.25*len(index1)), replace = False)
np.random.seed(seed)
index2 = np.random.choice(index2, size=int(0.5*len(index2)), replace = False)
indexn = np.where(train_label > 2)[0]
index = np.concatenate((index0,index1,index2,indexn), axis=None)
    
train_data = train_data[index]
train_label = train_label[index]
#unique, counts = np.unique(train_label, return_counts=True)


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

#[739204, 465439, 422277, 422674, 622298]
#[0.27665939, 0.17419828, 0.15804419, 0.15819277, 0.23290537]

print('prep complete, now fitting')
rf.fit(train_data, train_label)    
save = 'data/fitted_models/rf_model_window_set_depth_10.sav'
pickle.dump(rf, open(save, 'wb'))

finish = datetime.datetime.now()
print('finished at: ', finish)
print('took: ', finish-start)