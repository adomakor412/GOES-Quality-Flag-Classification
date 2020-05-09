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

rf = RF(n_estimators=0, criterion='gini', max_depth=10, random_state=42, warm_start=True)

for file in train:
    with h5py.File(file, 'r') as f:
        rf.n_estimators += 1
        label = f['label'][()]
        data = f['data'][()]
        rf.fit(data, label)
    print("file ", file, "done")
        
save = 'rf_model_depth10.sav'
pickle.dump(rf, open(save, 'wb'))
