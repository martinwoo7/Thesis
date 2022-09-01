import pandas as pd
import numpy as np



def get_maxmin(csv_path):
    mycsv = pd.read_csv(csv_path, encoding='cp1252',dtype='float64')
    mydata = mycsv.to_numpy()
    
    # remove labels
    mydata2 = mydata[:,1:]
    mydata3 = mydata2[:-1,:]

    # get acc only
    mydata4 = mydata3[:,0:300]
    print("max acc is", mydata4.max())
    print("min acc is", mydata4.min())

    # get gyro only
    mydata5 = mydata3[:,300:600]
    print("max gyro is", mydata5.max())
    print("min gyro is", mydata5.min())


def acc_norm(old):
    acc_max = 20
    acc_min = -20
    return int(((old - acc_min) / (acc_max - acc_min))*255)

def gyro_norm(old):
    gyro_max = 180
    gyro_min = -180
    return int(((old - gyro_min) / (gyro_max - gyro_min))*255)

def ori_norm(old):
    ori_max = 360
    ori_min = -360
    return int(((old - ori_min) / (ori_max - ori_min))*255)


def norm(csv_path):
    mycsv = pd.read_csv(csv_path, header = None)
    mydata = mycsv.to_numpy()
    original_shape = mydata.shape
    row, col = mydata.shape
    normed_data = np.zeros(shape=original_shape,dtype=np.int)




