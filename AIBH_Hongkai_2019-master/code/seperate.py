import numpy as np
import pandas as pd
import math
import os

'''
seperate a dataset into testing and training 
'''
def seperate(dir_path, train_output_path, test_output_path):
    colNum = 973
    # initialzation
    temp = np.zeros((1,colNum))

    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        # read the file
        csv_file = pd.read_csv(file_path, header = None)
        # convert it to an numpy array
        data = csv_file.to_numpy()
        temp = np.concatenate((temp, data), axis=0)
    
    temp = temp[1:,:]

    # random shuffle
    np.random.shuffle(temp)

    row, col = temp.shape

    train_row = int(row*0.8)

    train_set, test_set = temp[:train_row,:], temp[train_row:,:]

    np.savetxt(train_output_path, train_set,fmt='%i', delimiter=",")
    print("!complete!" + train_output_path)
    np.savetxt(test_output_path, test_set,fmt='%i', delimiter=",")
    print("!complete!" + test_output_path)

'''
merge all seperated datasets into two files
'''
def merge_csv(dir_path,output_path):
    colNum = 973
    # initialzation
    temp = np.zeros((1,colNum))

    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        # read the file
        csv_file = pd.read_csv(file_path, header = None)
        # convert it to an numpy array
        data = csv_file.to_numpy()
        temp = np.concatenate((temp, data), axis=0)
    
    temp = temp[1:,:]
    np.random.shuffle(temp)

    np.savetxt(output_path, temp,fmt='%i', delimiter=",")
    print("!complete!" + output_path)

dir_path = "processed"
train_output = "merged/train.csv"
test_output = "merged/test.csv"
seperate(dir_path, train_output, test_output)

        
        

