import numpy as np
import os
import h5py
import shutil
import requests

# PATH TO STORE THE TRAINING SAMPLES
outpath = '/Users/aleex/DSE/Machine Learning/Project/label/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

def genLabel(goes16, goes17):
    n1 = len(goes16)*len(goes16[0])
    n2 = len(goes17)*len(goes17[0])

    std1 = np.std(goes16)
    std2 = np.std(goes17)

    diff = goes16 - goes17
    diff_std = np.std(diff)

    delta = abs(goes16 - goes17)
    label = delta//diff_std
    label = label.astype(int)

    return label

def writeH5(data, label, uniTime):
    coords = np.array(data).astype(np.float)
    shape = coords.shape
    hdf5_path = outpath + uniTime + '.h5'
    with h5py.File(hdf5_path, mode='w') as f:
        d = f.create_dataset('/data', data = coords)
        l = f.create_dataset('/label', data = label)

#ROOT URL OF THE NPY FILES
rootURL = "https://gitlab.com/adomakor412/goesdata/-/raw/master/fall_npy/"
for idx in range(221, 238):
    for band in range(7, 9):
        for h in range(0, 24):
            for m in range(0, 6):
                timeID = f"s2019{str(idx)}{str(h).zfill(2)}{str(m*10).zfill(2)}"
                goes16_url = rootURL + f"{str(idx)}/OR_ABI-L1b-RadF-M6C{str(band).zfill(2)}_G16_{timeID}"
                goes17_url = rootURL + f"{str(idx)}/OR_ABI-L1b-RadF-M6C{str(band).zfill(2)}_G17_{timeID}"
                try:
                    response16 = requests.get(goes16_url, stream=True)
                    response17 = requests.get(goes17_url, stream=True)
                    print(timeID)
                    with open('temp16.npy', 'wb') as temp16:
                        shutil.copyfileobj(response16.raw, temp16)
                    with open('temp17.npy', 'wb') as temp17:
                        shutil.copyfileobj(response17.raw, temp17)
                    data16 = np.load('temp16.npy')
                    data17 = np.load('temp17.npy')
                    labels = genLabel(data16, data17)
                    writeH5(data17, labels, timeID)
                except ValueError:
                    print("url doesn't exist for " + str(goes16_url))
