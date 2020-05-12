from __future__ import print_function 
import numpy as np
import os
import os.path as op
import h5py
import shutil
import requests
import statistics

outpath = 'data_window/'

def createData(outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    def genLabel(g16, g17):
        n16 = g16.shape[0] * g16.shape[1]
        n17 = g17.shape[0] * g17.shape[1]
        
        diff_std = np.std(g16 - g17)
        abs_diff = abs(g16 - g17)
        
        label = abs_diff/diff_std
        label = abs(label-1)//1.5
        
        return label.astype(int)

    def writeH5(data, label, uniTime):
        coords = np.array(data).astype(np.float)
        shape = coords.shape
        hdf5_path = outpath + uniTime + '.h5'
        with h5py.File(hdf5_path, mode='w') as f:
            d = f.create_dataset('/data', data = coords)
            l = f.create_dataset('/label', data = label)
            
            
    def gen_windows(data, label):
        row, col, d = data.shape
        data_w = np.empty(((row-2)*(col-2), 36), float)
        label_w = np.empty(((row-2)*(col-2), 1), int)
        
        counter = 0
        for i in range(1,row-1):
            for j in range(1,col-1):
                data_w[counter] = data[(i-1):(i+2), (j-1):(j+2)].flatten()
                label_w[counter] = label[i,j]
                counter += 1
        return (data_w, label_w)


    #ROOT URL OF THE NPY FILES
    rootURL = "https://gitlab.com/adomakor412/goesdata/-/raw/master/spring_npy/"
    skip = False
    for idx in range(98, 109):
        for h in range(0, 24):
            for m in range(0, 6):
                data = np.empty((1001,401,0), float)
                label = np.empty((1001,401,0), int)
                timeID = "s2019{}{}{}".format(str(idx).zfill(3), str(h).zfill(2), str(m*10).zfill(2))
                for band in range(7, 11):
                    with open('log3', 'a') as log:
                        #timeID = "s2019{}{}{}".format(str(idx).zfill(3), str(h).zfill(2), str(m*10).zfill(2))

                        goes16_url = rootURL + "{}/OR_ABI-L1b-RadF-M6C{}_G16_{}".format(str(idx).zfill(3), str(band).zfill(2), timeID)
                        goes17_url = rootURL + "{}/OR_ABI-L1b-RadF-M6C{}_G17_{}".format(str(idx).zfill(3), str(band).zfill(2), timeID)

                        try:
                            response16 = requests.get(goes16_url, stream=True)
                            response17 = requests.get(goes17_url, stream=True)
                            
                            with open('temp16.npy', 'wb') as temp16:
                                shutil.copyfileobj(response16.raw, temp16)
                            with open('temp17.npy', 'wb') as temp17:
                                shutil.copyfileobj(response17.raw, temp17)

                            data16 = np.load('temp16.npy').reshape((1001, 401, 1))
                            data17 = np.load('temp17.npy').reshape((1001, 401, 1))

                            labels = genLabel(data16, data17)                                                                                                                                                      
                            data = np.append(data, data17, axis = 2)

                            label = np.append(label, labels, axis = 2)

                        except Exception as e:
                            skip = True
                            print("", file=log)
                            print(goes16_url, response16.status_code, file=log)
                            print(goes17_url, response17.status_code, file=log)
                            print(e, file=log)
                            break
                                                
                if skip:
                    skip=False
                    continue
                
                print(timeID)
                vote = np.round(np.mean(label, axis=2)).astype('int')
                data_w, label_w = gen_windows(data, vote)
                label_w = label_w.reshape((-1))
                writeH5(data_w, label_w, timeID)
         
createData(outpath)
                            