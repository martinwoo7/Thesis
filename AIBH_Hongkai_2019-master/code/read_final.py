import numpy as np
import pandas as pd
import math
import os
import normalization as nm

'''
given a path, read a single csv file into two numpy arrays
'''
def get_data_single(file_path):
    data_file = pd.read_csv(file_path)
    # convert it to an numpy array
    data = data_file.to_numpy()

    # X --> only original datasets
    # [[timestamp, rel_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, azimuth, pitch, roll],...]
    X = data[:, :-1]
    # Y --> only labels
    # [[label],...]
    Y = data[:, -1]

    return X, Y


'''
given a folder, read all the csv files into numpy arrays
'''
def get_data_folder(dir_path):
    X = np.zeros(shape = [1,11])
    Y = np.zeros(shape = [1,1])
    # get each file's name
    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        # get this file's X and Y
        temp_X, temp_Y = get_data_single(file_path)
        # merge them into the main numpy array
        ##### print("file name is "+ file_name)
        X = np.concatenate((X,temp_X),axis=0)
        ##### print("file name is "+ file_name)
        Y = np.concatenate((Y,temp_Y),axis=None)
    
    X = X[1:,]
    Y = Y[1:,]

    return X, Y

'''
help function
'''
def window(np_array, window_size, overlap):
    sh = (np_array.size - window_size + 1, window_size)
    st = np_array.strides * 2
    try:
        view = np.lib.stride_tricks.as_strided(np_array, strides = st, shape = sh)[0::overlap]
    except:
        view = np.zeros(shape = [1,100])
        print("An exception occurred")
    

    return view

'''
re-compute new labels
'''
def label_assignment(Y, real_time, new_dataset_size, frequency, resample_size, overlap, last_x_cordinates):
    x_cordinates_count = 0
    new_labels = np.zeros((new_dataset_size, 1))
    Label = {'STD':1, 'WAL':2,'JOG':3,'JUM':4,'STU':5,'STN':6,'SCH':7,'SIT':8,'CHU':9,'CSI':10,'CSO':11,'LYI':12,'FOL':20,'FKL':21,'BSC':22,'SDL':23}

    for count in range(new_dataset_size):
        start = x_cordinates_count
        end = x_cordinates_count + resample_size*frequency
        # get the index
        start_idx = (np.abs(real_time - start)).argmin()
        end_idx = (np.abs(real_time - end)).argmin()
        # find the most frequent label
        (values,counts) = np.unique(Y[start_idx:end_idx, ],return_counts=True)
        most_frequent_label = values[np.argmax(counts)]
        # save it
        most_frequent_label_int = Label[most_frequent_label]
        new_labels[count,] = most_frequent_label_int
        # update count
        x_cordinates_count += resample_size*frequency*overlap
        count += 1
    
    return new_labels


'''
Pre-processing
'''
def pre_processing(file_path):
    window_size = 1000
    frequency = 0.01
    resample_size = 108
    overlap = 0.5
    overlap_size = int(resample_size*overlap)
    # [[timestamp, rel_time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, azimuth, pitch, roll],...]
    X, Y = get_data_single(file_path)
    X_len, X_width = X.shape

    # get each sequece and change the shape from (n,1) to (n,)
    real_time = X[:, 1:2].reshape(X_len).astype('float64')
    acc_x = X[:, 2:3].reshape(X_len).astype('float64')
    acc_y = X[:, 3:4].reshape(X_len).astype('float64')
    acc_z = X[:, 4:5].reshape(X_len).astype('float64')
    gyro_x = X[:, 5:6].reshape(X_len).astype('float64')
    gyro_y = X[:, 6:7].reshape(X_len).astype('float64')
    gyro_z = X[:, 7:8].reshape(X_len).astype('float64')
    ori_x = X[:, 8:9].reshape(X_len).astype('float64')
    ori_y = X[:, 9:10].reshape(X_len).astype('float64')
    ori_z = X[:, 10:11].reshape(X_len).astype('float64')




    # get the very last time of this dataset
    last_time = X[X_len-1:,1:2].item(0) #float
    # round it to 0.05*n
    last_x_cordinates = round(last_time - (last_time % frequency), 2)

    # get total num of resampled data, discard remainder
    total_num = math.ceil(last_x_cordinates / frequency)
    total_num = total_num - (total_num % resample_size)

    # x-cordinates 20hz --> 0.05 sec each
    # [0.05,0.10,0.15...]
    
    x_cordinates = np.linspace(0, last_x_cordinates, num= total_num )


    # use linear interpolation to resample the data
    itp_acc_x = np.interp(x_cordinates, real_time, acc_x)
    itp_acc_y = np.interp(x_cordinates, real_time, acc_y)
    itp_acc_z = np.interp(x_cordinates, real_time, acc_z)
    itp_gyro_x = np.interp(x_cordinates, real_time, gyro_x)
    itp_gyro_y = np.interp(x_cordinates, real_time, gyro_y)
    itp_gyro_z = np.interp(x_cordinates, real_time, gyro_z)
    itp_ori_x = np.interp(x_cordinates, real_time, ori_x)
    itp_ori_y = np.interp(x_cordinates, real_time, ori_y)
    itp_ori_z = np.interp(x_cordinates, real_time, ori_z)

    

    # construct the result

    # calculate the size of new dataset
    new_dataset_size = round(1 + (total_num - resample_size) / (resample_size * overlap))

    new_labels = label_assignment(Y, real_time, new_dataset_size, frequency, resample_size, overlap, last_x_cordinates)
    #print("new_dataset_size", new_dataset_size)

    # Use help function to reshape the data with a fixed window size and a fixed overlap rate
    new_acc_x = window(itp_acc_x, resample_size, overlap_size)
    new_acc_y = window(itp_acc_y, resample_size, overlap_size)
    new_acc_z = window(itp_acc_z, resample_size, overlap_size)
    new_gyro_x = window(itp_gyro_x, resample_size, overlap_size)
    new_gyro_y = window(itp_gyro_y, resample_size, overlap_size)
    new_gyro_z = window(itp_gyro_z, resample_size, overlap_size)
    new_ori_x = window(itp_ori_x, resample_size, overlap_size)
    new_ori_y = window(itp_ori_y, resample_size, overlap_size)
    new_ori_z = window(itp_ori_z, resample_size, overlap_size)
    #print("new_acc_x", new_acc_x.shape)

    # Normalization
    normed_acc_x = (np.around((np.where(True, ((new_acc_x + 20) / (40) * 255), 0)))).astype(int)
    normed_acc_y = (np.around((np.where(True, ((new_acc_y + 20) / (40) * 255), 0)))).astype(int)
    normed_acc_z = (np.around((np.where(True, ((new_acc_z + 20) / (40) * 255), 0)))).astype(int)
    normed_gyro_x = (np.around((np.where(True, ((new_gyro_x + 180) / (360) * 255), 0)))).astype(int)
    normed_gyro_y = (np.around((np.where(True, ((new_gyro_y + 180) / (360) * 255), 0)))).astype(int)
    normed_gyro_z = (np.around((np.where(True, ((new_gyro_z + 180) / (360) * 255), 0)))).astype(int)
    normed_ori_x = (np.around((np.where(True, ((new_ori_x + 360) / (720) * 255), 0)))).astype(int)
    normed_ori_y = (np.around((np.where(True, ((new_ori_y + 360) / (720) * 255), 0)))).astype(int)
    normed_ori_z = (np.around((np.where(True, ((new_ori_z + 360) / (720) * 255), 0)))).astype(int)


    return normed_acc_x, normed_acc_y, normed_acc_z, normed_gyro_x, normed_gyro_y, normed_gyro_z, normed_ori_x, normed_ori_y, normed_ori_z, new_labels


'''
output new datasets in csv format
'''
def writeCSV(file_path, output_path):
    normed_acc_x, normed_acc_y, normed_acc_z, normed_gyro_x, normed_gyro_y, normed_gyro_z, normed_ori_x, normed_ori_y, normed_ori_z, new_labels = pre_processing(file_path)

    '''
    labels_X, labels_Y = normed_acc_x.shape
    labels = (np.full((labels_X, 1), label)).astype(int)
    '''

    # merge them together
    #print(labels.shape)
    #print(new_acc_x.shape)
    step1 = np.concatenate((new_labels, normed_acc_x), axis=1)
    step2 = np.concatenate((step1, normed_gyro_x), axis=1)
    step3 = np.concatenate((step2, normed_ori_x), axis=1)
    step4 = np.concatenate((step3, normed_acc_y), axis=1)
    step5 = np.concatenate((step4, normed_gyro_y), axis=1)
    step6 = np.concatenate((step5, normed_ori_y), axis=1)
    step7 = np.concatenate((step6, normed_acc_z), axis=1)
    step8 = np.concatenate((step7, normed_gyro_z), axis=1)
    step9 = np.concatenate((step8, normed_ori_z), axis=1)
    #print(step6.shape)

    np.savetxt(output_path, step9.astype(int),fmt='%i', delimiter=",")
    print(output_path)

'''
main function
'''
def final_processing(dir_path,output_dir):
    count = 1
    for file_name in os.listdir(dir_path):
        file_path = dir_path + '/' + file_name
        output_path = output_dir + file_name
        writeCSV(file_path, output_path)
        count += 1

activity_path = "data/Annotated Data/activities"
output_dir = "processed/"
for activity in os.listdir(activity_path):
    activity_file_path = activity_path + '/' + activity
    final_processing(activity_file_path, output_dir)



