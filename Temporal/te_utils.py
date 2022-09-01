import os
import time
import h5py
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import kurtosis, skew

from sklearn.pipeline import make_pipeline
from tensorflow.keras import utils
from sklearn import metrics, model_selection, preprocessing

MIN_ACCEL = -20
MAX_ACCEL = 20
MIN_GYRO = -180
MAX_GYRO = 180
MIN_ORIEN = -360
MAX_ORIEN = 360

def fuse(data=None, labs=None, mode=2):
    '''
    This function tries to fuse the sensor information by creating new ones.
    I'm not exactly sure how many features I should create.
    This is for 1a: manual extraction. I want to provide three methods of implementing as according to what I wrote in the thesis
    1. Use only new features after condensing
    2. append new features and reduce everything to only 1 feature (thru temporal fusion)
    3. Complementary filter and/or Kalman filter

    data columns goes: acc x y z, magne x y z, gyro x y z
    for 1) we will use: mean, median, standard deviation, kurtosis, variance, and skewness
    this is for EACH axis, meaning acc x will get these features, and so will acc y and acc z
    '''
    if (mode == 1):
        reduced = []
        reduced_labs = []

        # step = 100
        # for i in range(0, data.shape[0]-step, step):
        #     subset = data[i:i+step * 2]
        #     sublab = labs[i:i+step * 2]

        step = 200
        for i in range(0, data.shape[0], step):
            subset = data[i:i+step]
            sublab = labs[i:i+step]

            u, c = np.unique(sublab, return_counts=True)
            reduced_labs.append(u[c == c.max()][0])

            row = []
            slope_array = []
            for j in range(data.shape[1]):
                # iterate through all columns
                arr = [x[j] for x in subset]
                mean_array = np.mean(arr)
                median_array = np.median(arr)
                std_array = np.std(arr)
                kurt_array = kurtosis(arr)
                var_array = np.var(arr)
                skew_array = skew(arr)
                min_array = min(arr)
                max_array = max(arr)
                # hmean_array = hmean(arr)
                zrc_array = sum((np.array(arr)[:-1] * np.array(arr)[1:] < 0)) / len(arr)
                total_array = [mean_array, median_array, std_array, kurt_array, var_array, skew_array, min_array, max_array, zrc_array]
                
                if len(row) == 0:
                    row = total_array
                    slope_array = [min_array, max_array]
                else:
                    row = np.append(row, total_array)
                    slope_array = np.vstack((slope_array, [min_array, max_array]))
            
            summ = 0
            for axi in slope_array:
                summ += (axi[1] - axi[0])**2
            
            slope = math.sqrt(summ)
            row = np.append(row, slope)

            if len(reduced) == 0:
                reduced = row
            else:
                reduced = np.vstack((reduced, row))

        reduced_labs = np.array(reduced_labs).reshape(-1, 1)
        return np.array(reduced), reduced_labs

    elif (mode == 2):
        dataset_path = os.path.join(os.getcwd(), "data/Annotated Data/activities")

        try: 
            training_data = getDataset("stacked_training_data.hdf5", "stacked_training_dataset")
            training_labels = getDataset("stacked_training_labels.hdf5", "stacked_training_labels")

        except OSError:
            print("Stacked Training dataset not found! Starting pre-processing")

            training_data = []
            training_labels = []
            instantiated = False
            for folder in os.listdir(os.fsencode(dataset_path)):
                # open the folder for each activity
                folderName = os.fsdecode(folder)
                activity_path = dataset_path + '/' + folderName
                print("Loading in %s data" % folderName)
                
                for count, activity in enumerate(os.listdir(os.fsencode(activity_path))):
                    # loading in individual activity files
                    activity_file_path = os.path.join(activity_path + '/' + os.fsdecode(activity))
                    activity_file = pd.read_csv(activity_file_path)
                    
                    categorical = {"label": {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "STU": 4, "STN": 5, "SCH": 6, "SIT": 7, "CHU": 8, "CSI": 9, "CSO": 10, "LYI": 11, "FOL": 12, "FKL": 13, "BSC": 14, "SDL": 15}}
                    activity_file = activity_file.replace(categorical)
                    label_list = (activity_file.loc[:, "label"]).to_numpy()
                    activity_file = activity_file.drop(columns=["timestamp", "label", "rel_time"])
                    # activity_file = preprocessing.StandardScaler().fit_transform(activity_file)
                    data_list = activity_file.to_numpy()

                    changed_data, changed_labels = fuse(data_list, label_list, 1) # returns the reduced manual data)
                    if ( not instantiated ):
                        training_data = changed_data
                        training_labels = changed_labels
                        instantiated = True
                    else:
                        training_data = np.vstack((training_data, changed_data))
                        training_labels = np.vstack((training_labels, changed_labels))
                    
            # max_activity = max(activity_count, key=lambda k: activity_count[k])
            training_labels = utils.to_categorical(training_labels, num_classes=16)

            print(training_data.shape)
            print(training_labels.shape)

            x_train, x_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, test_size=0.2, random_state=42)
            print(x_train.shape, y_train.shape)
            print(x_test.shape, y_test.shape)

            print("Saving dataset")
            # saving training dataset and labels
            training_location = os.getcwd()
            f1 = h5py.File("stacked_training_data.hdf5", "w")
            f1.create_dataset("stacked_training_dataset", data=x_train)
            print("Saved training_data at %s" %(training_location))
            f1.close()

            f2 = h5py.File("stacked_training_labels.hdf5", "w")
            f2.create_dataset("stacked_training_labels", data=y_train)
            print("Saved training_labels at %s" %(training_location))
            f2.close()

            f3 = h5py.File("stacked_testing_data.hdf5", "w")
            f3.create_dataset("stacked_testing_dataset", data=x_test)
            print("Saved testing_data at %s" %(training_location))
            f3.close()

            f4 = h5py.File("stacked_testing_labels.hdf5", "w")
            f4.create_dataset("stacked_testing_labels", data=y_test)
            print("Saved testing_labels at %s" %(training_location))
            f4.close()
        else:
            print("Training data found!")

        return x_train, x_test, y_train, y_test

    elif (mode == 3):
        print("creating time sequences")
        dataset_path = os.path.join(os.getcwd(), "data/Annotated Data/activities")

        try:
            data = getDataset("timeseries_data.hdf5", "timeseries_data")
            labels = getDataset("timeseries_labels.hdf5", "timeseries_labels")

        except OSError:
            print("Time series dataset not found! Starting pre-processing")
            data = []
            labels = []

            for folder in os.listdir(os.fsencode(dataset_path)):
                folderName = os.fsdecode(folder)
                activity_path = dataset_path + '/' + folderName
                print("Loading in %s data" % folderName)

                for count, activity in enumerate(os.listdir(os.fsencode(activity_path))):
                    if count % 3 != 0:
                        continue
                    activity_file_path = os.path.join(activity_path + '/' + os.fsdecode(activity))
                    activity_file = pd.read_csv(activity_file_path)
                    categorical = {"label": {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "STU": 4, "STN": 5, "SCH": 6, "SIT": 7, "CHU": 8, "CSI": 9, "CSO": 10, "LYI": 11, "FOL": 12, "FKL": 13, "BSC": 14, "SDL": 15}}
                    activity_file = activity_file.replace(categorical)
                    label_list = (activity_file.loc[:, "label"]).to_numpy()
                    activity_file = activity_file.drop(columns=["timestamp", "label", "rel_time"])
                    data_list = activity_file.to_numpy()

                    changed_data, changed_labels = fuse(data_list, label_list, 1)
                
                    changed_data = changed_data.tolist()
                    # changed_labels = changed_labels.tolist()
                
                    data.append(changed_data)
                    # labels.append(changed_labels)
                    labels.append(folderName)
        return data, labels

def train_generator(scaler=None, state=42, singular=False):
    dataset_path = os.path.join(os.getcwd(), "data/Annotated Data/activities")
    categorical = {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "STU": 4, "STN": 5, "SCH": 6, "SIT": 7, "CHU": 8, "CSI": 9, "CSO": 10, "LYI": 11, "FOL": 12, "FKL": 13, "BSC": 14, "SDL": 15}
    if len(os.listdir("segmented")) == 0:
        iterated = False
    else:
        iterated = True

    while True:
        if (not iterated):
            for folder in os.listdir(os.fsencode(dataset_path)):
                # open the folder for each activity
                folderName = os.fsdecode(folder)
                
                activity_path = dataset_path + '/' + folderName
                print("Loading in %s data" % folderName)

                action, label = [], []
                instantiated = False

                for count, activity in enumerate(os.listdir(os.fsencode(activity_path))):
                    # loading in individual activity files
                    if count % 3 != 0:
                        continue

                    activity_file_path = os.path.join(activity_path + '/' + os.fsdecode(activity))
                    activity_file = pd.read_csv(activity_file_path)
                    # activity_file = activity_file.replace(categorical)
                    label_list = (activity_file.loc[:, "label"]).to_numpy()
                    activity_file = activity_file.drop(columns=["timestamp", "label", "rel_time"])
                    data_list = activity_file.to_numpy()

                    changed_data, changed_labels = fuse(data_list, label_list, 1) # returns the reduced manual data)
                    
                    # we're going to normalize here just to keep things clean. Scaler fit on entire dataset
                    changed_data = scaler.transform(changed_data)
                    # changed_data = preprocessing.StandardScaler().fit_transform(changed_data)

                    # changed_labels = changed_labels.reshape(1, changed_labels.shape[0], changed_labels.shape[1])
                    # changed_labels = utils.to_categorical(changed_labels, num_classes=len(categories))
                    
                    if (not instantiated):
                        action = changed_data.reshape(1, changed_data.shape[0], changed_data.shape[1])
                        instantiated = True
                        label.append(categorical[folderName])
                    else:
                        if action.shape[1] > changed_data.shape[0]:
                        # Repeat last vectors of changed_data by the difference
                            remainder = action.shape[1] % changed_data.shape[0]
                            temp_tile = np.tile(changed_data[-1], remainder).reshape(-1, changed_data.shape[1])
                            changed_data = np.vstack((changed_data, temp_tile))
            
                        elif action.shape[1] < changed_data.shape[0]:
                            # remove the remainder from changed_data to fit
                            remainder = changed_data.shape[0] % action.shape[1]
                            for _ in range(remainder):
                                changed_data = np.delete(changed_data, -1, 0)

                        changed_data = changed_data.reshape(1, changed_data.shape[0], changed_data.shape[1])
                        action = np.vstack((action, changed_data))
                        label.append(categorical[folderName])

                # print(action.shape)
                # print("Skipped %i %s files" % (skipped, folderName))
                if (folderName == "WAL"):
                    iterated = True

                length = 16

                remainder = action.shape[1] % length
                if remainder != 0: 
                    temp = action[-1][-1].reshape(-1, action.shape[2])
                    temp_tile = np.tile(temp, (-action.shape[1]) % length).reshape(-1, action.shape[2])
                    temp_tile = np.tile(temp_tile, action.shape[0]).reshape(-1, temp_tile.shape[0], temp_tile.shape[1])
                    action = np.append(action, temp_tile, axis=1) 
                
                n_steps, n_length = action.shape[1] // length, length
                action = action.reshape(action.shape[0], n_steps, n_length, action.shape[2])

                changed_labels = utils.to_categorical(np.array(label).reshape(-1, 1), num_classes=16)

                print(action.shape)
                print(changed_labels.shape)

                # assert False
                f1 = h5py.File("segmented/" + folderName + ".hdf5", "w")
                f1.create_dataset(folderName, data=action)
                f1.close()
                
                f2 = h5py.File("segmented/" + folderName +"_labels.hdf5", "w")
                f2.create_dataset(folderName + "_labels", data=changed_labels)
                f2.close()
                
                yield action, changed_labels
        else:
            # load from data
            for folder in os.listdir(os.fsencode(dataset_path)):
                folderName = os.fsdecode(folder)
                # print(folderName)
                # activity_path = dataset_path + '/' + folderName

                action = getDataset("segmented/" + folderName + ".hdf5", folderName)
                label = getDataset("segmented/" + folderName + "_labels.hdf5", folderName+"_labels")

                x_train, x_test, y_train, y_test = model_selection.train_test_split(action, label, test_size=0.3, random_state=state)
                x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
                if singular:
                    yield x_train, y_train
                else:
                    yield [x_train, x_train, x_train], y_train

segmented = "datasets/segmented_test2"
def train_generator2(scaler=None, state=42, encode=False, group=""):
    if group == "":
        print("No group specified")
        assert False

    dataset_path = os.path.join(os.getcwd(), "data/Annotated Data/activities")
    categorical = {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "STU": 4, "STN": 5, "SCH": 6, "SIT": 7, "CHU": 8, "CSI": 9, "CSO": 10, "LYI": 11, "FOL": 12, "FKL": 13, "BSC": 14, "SDL": 15}

    if len(os.listdir(segmented + '/train')) == 0:
        iterated = False
    else:
        iterated = True

    while True:
        if not iterated:
            for folder in os.listdir(os.fsencode(dataset_path)):
                folderName = os.fsdecode(folder)
                if folderName != "WAL":
                    print("skipped ", folderName)
                    continue
                activity_path = dataset_path + '/' + folderName
                print("loading in %s data" % folderName)

                action, label, label_final = [], [], None
                instantiated = False

                for count, activity in enumerate(os.listdir(os.fsencode(activity_path))):
                    # if count % 3 != 0:
                    #     continue
                    
                    activity_file_path = os.path.join(activity_path + '/' + os.fsdecode(activity))
                    activity_file = pd.read_csv(activity_file_path)

                    # Do normalization
                    # activity_file[['acc_x','acc_y','acc_z']] = (activity_file[['acc_x','acc_y','acc_z']]-MIN_ACCEL)/(MAX_ACCEL-MIN_ACCEL)
                    # activity_file[['gyro_x','gyro_y','gyro_z']] = (activity_file[['gyro_x','gyro_y','gyro_z']]-MIN_GYRO)/(MAX_GYRO-MIN_GYRO)
                    # activity_file[['azimuth','pitch','roll']] = (activity_file[['azimuth','pitch','roll']]-MIN_ORIEN)/(MAX_ORIEN-MIN_ORIEN) 
                    
                    label_list = activity_file.loc[:, "label"]
                    label_list = label_list.replace(categorical).to_numpy()
                    activity_file = activity_file.drop(columns=["timestamp", "label", "rel_time"])

                    data_list = activity_file.to_numpy()

                    # This should return (n, 82) where n is the len(file) / 200 and 82 are the new features
                    # Also returns (n, 1) for labels
                    # Both are np arrays
                    changed_data, changed_labels = fuse(data_list, label_list, 1)
                    # changed_data = scaler.transform(changed_data)
                    changed_labels = utils.to_categorical(changed_labels, 16)

                    # changed_data = data_list
                    # changed_labels = utils.to_categorical(label_list, 16)

                    if (not instantiated):
                        action = changed_data.reshape(1, changed_data.shape[0], changed_data.shape[1])
                        instantiated = True
                        label = changed_labels.reshape(1, changed_labels.shape[0], changed_labels.shape[1])

                        label_final = np.array([categorical[folderName]])
                    else:
                        if action.shape[1] > changed_data.shape[0]:
                        # Repeat last vectors of changed_data by the difference
                            remainder = action.shape[1] % changed_data.shape[0]
                            temp_tile = np.tile(changed_data[-1], remainder).reshape(-1, changed_data.shape[1])
                            changed_data = np.vstack((changed_data, temp_tile))

                            label_tile = np.tile(changed_labels[-1], remainder).reshape(-1, changed_labels.shape[1])
                            changed_labels = np.vstack((changed_labels, label_tile))
            
                        elif action.shape[1] < changed_data.shape[0]:
                            # remove the remainder from changed_data to fit
                            remainder = changed_data.shape[0] % action.shape[1]
                            for _ in range(remainder):
                                changed_data = np.delete(changed_data, -1, 0)
                                changed_labels = np.delete(changed_labels, -1, 0)

                        changed_data = changed_data.reshape(1, changed_data.shape[0], changed_data.shape[1])
                        action = np.vstack((action, changed_data))

                        changed_labels = changed_labels.reshape(1, changed_labels.shape[0], changed_labels.shape[1])
                        label = np.vstack((label, changed_labels))

                        label_final = np.vstack((label_final, categorical[folderName]))
                        
                    # At this point the shape of labels should be (1, n, 16) and shape of data is (1, n, 82)
                label_final = utils.to_categorical(label_final, num_classes=16)
                print(folderName, action.shape, label_final.shape)
                x_train, x_test, y_train, y_test = model_selection.train_test_split(action, label_final, test_size=0.2, random_state=state)
                print(x_train.shape, y_train.shape)
                print(x_test.shape, y_test.shape)

                if folderName == "WAL":
                    iterated = True
                
                f1 = h5py.File(segmented + "/train/" + folderName + ".hdf5", "w")
                f1.create_dataset(folderName, data=x_train)
                f1.close()
                
                f2 = h5py.File(segmented + "/train/" + folderName +"_labels.hdf5", "w")
                f2.create_dataset(folderName + "_labels", data=y_train)
                f2.close()

                f3 = h5py.File(segmented + "/test/" + folderName + ".hdf5", "w")
                f3.create_dataset(folderName, data=x_test)
                f3.close()
                
                f4 = h5py.File(segmented + "/test/" + folderName +"_labels.hdf5", "w")
                f4.create_dataset(folderName + "_labels", data=y_test)
                f4.close()

                yield action, label_final
        else:
            for folder in os.listdir(os.fsencode(dataset_path)):
                folderName = os.fsdecode(folder)
                action = getDataset(segmented + "/" + group + '/' + folderName + ".hdf5", folderName)
                label = getDataset(segmented + "/" + group + '/'+  folderName + "_labels.hdf5", folderName+"_labels")
                # print(action.shape, label.shape)
                orig_shape = action.shape
                action = scaler.transform(action.reshape(-1, action.shape[-1]))
                action = action.reshape(orig_shape)
                # Splitting between validation data
                x_train, x_test, y_train, y_test = model_selection.train_test_split(action, label, test_size=0.2, random_state=state)
                if encode:
                    yield x_train, x_train
                else:
                    yield x_train, y_train

def validate_generator(state, singular=False):
    dataset_path = os.path.join(os.getcwd(), "data/Annotated Data/activities")
    while True:
        for folder in os.listdir(os.fsencode(dataset_path)):
            folderName = os.fsdecode(folder)
            # activity_path = dataset_path + '/' + folderName

            action = getDataset("segmented/" + folderName + ".hdf5", folderName)
            label = getDataset("segmented/" + folderName + "_labels.hdf5", folderName+"_labels")

            x_train, x_test, y_train, y_test = model_selection.train_test_split(action, label, test_size=0.2, random_state=state)
        
            if singular:
                yield x_test, y_test
            else:
                yield [x_test, x_test, x_test], y_test

def validate_generator2(state, encode=False, group=""):
    if not group:
        print("No group specified")
        assert False

    dataset_path = os.path.join(os.getcwd(), "data/Annotated Data/activities")
    while True:
        for folder in os.listdir(os.fsencode(dataset_path)):
            folderName = os.fsdecode(folder)
            action = getDataset(segmented + "/" + group + '/' + folderName + ".hdf5", folderName)
            label = getDataset(segmented + "/" + group + '/' + folderName + "_labels.hdf5", folderName+"_labels")

            x_train, x_test, y_train, y_test = model_selection.train_test_split(action, label, test_size=0.2, random_state=state)
            
            if encode:
                yield x_test, x_test
            else:
                yield x_test, y_test


def bench_k_means(kmeans, name, data, labels):
    t0 = time.time()
    # estimator = make_pipeline(preproc.StandardScaler(), kmeans).fit(training_data)
    estimator = make_pipeline(kmeans).fit(data)

    fit_time = time.time() - t0
    results = [name, fit_time, estimator[-1].inertia_]
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.normalized_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))
    return estimator[-1].labels_

def getDataset(filename, dataset_name):
    hf = h5py.File(filename, "r")
    temp = np.array(hf.get(dataset_name))
    hf.close()
    return temp

def load_train(name):

    group = "_training"
    x = getDataset("datasets/" + name + "/train/" + name + group + "_data.hdf5", name+group+"_data")
    y = getDataset("datasets/" + name + "/train/" + name + group + "_labels.hdf5", name+group+"_labels")
    return x,y 

def load_test(name):
    group = "_testing"
    x = getDataset("datasets/" + name + "/test/" + name + group + "_data.hdf5", name+group+"_data")
    y = getDataset("datasets/" + name + "/test/" + name + group + "_labels.hdf5", name+group+"_labels")
    return x,y 