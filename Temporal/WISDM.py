import math
import os
os.environ["OMP_NUM_THREADS"] = "4"
import time
import h5py

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import pandas as pd
import tmap as tm
from scipy.spatial import distance as dst

from h5py._hl.files import File
from scipy.stats import hmean, kurtosis, skew
from faerun import Faerun
from flexible_clustering import FISHDBC 
from sklearn import metrics, model_selection
from sklearn import preprocessing as preproc
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from tensorflow.keras import (callbacks, datasets, layers, models, optimizers,
                              utils, preprocessing, regularizers)
from operator import itemgetter
from itertools import groupby


@nb.jit(forceobj=True)
def is_in_set_nb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, sep=',', lineterminator=';')
    return dataframe.values

def getDataset(filename, dataset_name):
    hf = h5py.File(filename, "r")
    temp = np.array(hf.get(dataset_name))
    hf.close()
    return temp

def fuse(data=None, labs=None, mode=2):
    if mode == 1:
        step = 200
        reduced = []
        reduced_labs = []
        for i in range(0, data.shape[0], step):
            # iterate through all data in step of 200
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

def getGroup(path, name):
    data_path = path + '/' + name
    loaded = None
    loaded_labs = []
    for file in os.listdir(os.fsencode(data_path)):
        # loaded = None
        # loaded_labs = []
        fileName = os.fsdecode(file)
        # print("Loading in ", fileName)

        accel_path = data_path + '/' + fileName
        gyro_path = accel_path.replace("accel", "gyro")

        accel_file = load_file(accel_path)
        gyro_file = load_file(gyro_path)

        # This returns a boolean array of timestamps of accel that exists in gyro

        timestamp_mask = is_in_set_nb(accel_file[:, 2], gyro_file[:, 2])
        timestamp_mask_gyro = is_in_set_nb(gyro_file[:, 2], accel_file[:, 2])
        accel_data = np.compress(timestamp_mask, accel_file, axis=0)
        gyro_data = np.compress(timestamp_mask_gyro, gyro_file, axis=0)
        phone_labels = np.compress(timestamp_mask, accel_file[:, 1], axis=0)

        accel_label_index = [sorted(set(itemgetter(0, -1)([i[0] for i in g]))) for _, g in groupby(enumerate(phone_labels), key=itemgetter(1))]
        # gyro_label_index = [sorted(set(itemgetter(0, -1)([i[0] for i in g]))) for _, g in groupby(enumerate(gyro_label), key=itemgetter(1))]

        # Loading in the watch data 
        watch_accel_path = accel_path.replace("phone", "watch")
        watch_gyro_path = watch_accel_path.replace("accel", "gyro")
        watch_accel_file = load_file(watch_accel_path)
        watch_gyro_file = load_file(watch_gyro_path)

        timestamp_mask = is_in_set_nb(watch_accel_file[:, 2], watch_gyro_file[:, 2])
        timestamp_mask_gyro = is_in_set_nb(watch_gyro_file[:, 2], watch_accel_file[:, 2])
        watch_accel_data = np.compress(timestamp_mask, watch_accel_file, axis=0)
        watch_gyro_data = np.compress(timestamp_mask_gyro, watch_gyro_file, axis=0)
        watch_labels = np.compress(timestamp_mask, watch_accel_file[:, 1], axis=0)

        watch_accel_label_index = [sorted(set(itemgetter(0, -1)([i[0] for i in g]))) for _, g in groupby(enumerate(watch_labels), key=itemgetter(1))]
        # watch_gyro_label_index = [sorted(set(itemgetter(0, -1)([i[0] for i in g]))) for _, g in groupby(enumerate(watch_gyro_label), key=itemgetter(1))]

        # if len(watch_accel_label_index) != len(watch_gyro_label_index):
        #     print("inconsistent watch labels index for ", fileName.replace("phone", "watch"))
        #     assert False

        if len(watch_accel_label_index) != len(accel_label_index):
            print("inconsistent lengths between watch and phone index for ", fileName, " and ", fileName.replace("phone", "watch"))
            assert False


        for index, data_group in enumerate(accel_label_index):
            # I'm assuming that both the watch and phone must have the same number of activities
            phone_time = np.hstack((accel_data[:, 2:3], gyro_data[:, 2:3]))
            watch_time = np.hstack((watch_accel_data[:, 2:3], watch_gyro_data[:, 2:3]))


            fused_phone_data = np.hstack((accel_data[:, 3:], gyro_data[:, 3:]))
            fused_watch_data = np.hstack((watch_accel_data[:, 3:], watch_gyro_data[:, 3:]))

            print(fused_phone_data.shape)
            print(fused_watch_data.shape)

            return fused_phone_data, phone_time, fused_watch_data, watch_time


            assert False
            reduced_phone_data, reduced_phone_labels = fuse(fused_phone_data, phone_labels, mode=1)
            reduced_watch_data, reduced_watch_labels = fuse(fused_watch_data, watch_labels, mode=1)

            print(reduced_phone_data.shape)
            print(reduced_watch_data.shape)
            assert False
            if reduced_data.shape[0] < 18:
                remainder = 18 - reduced_data.shape[0]
                last = reduced_data[-1]
                for _ in range(remainder):
                    reduced_data = np.vstack((reduced_data, last))

            elif reduced_data.shape[0] > 18:
                remainder = reduced_data.shape[0] % 18
                reduced_data = reduced_data[:-remainder]

            if loaded is None:
                loaded = reduced_data.reshape(-1, 18, reduced_data.shape[1])
                loaded_labs = loaded.shape[0] * [reduced_labels[0]]
            else:
                reduced_data = reduced_data.reshape(-1, 18, reduced_data.shape[1])
                loaded = np.vstack((loaded, reduced_data))
                loaded_labs += reduced_data.shape[0] * [reduced_labels[0]]
        # if loaded is None:
        #     continue
        # else:
        #     print(loaded.shape)
        #     print(np.array(loaded_labs).shape)
    return loaded, np.array(loaded_labs)

def distance(x, y):
    # return dst.euclidean(x, y)
    return dst.minkowski(x, y, 1)
    # return dst.cityblock(x, y)
    # return dtw(x, y, distance_only=True).distance

def visualize():
    assert False

def cluster(model, mode=1, data=None, labels=None):
    popped_model = models.Model(inputs=model.input, outputs=model.layers[-3].output)

if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), "wisdm/raw/phone")
    phone_data, phone_label, watch_data, watch_label = getGroup(dataset_path, "accel")


    # print("Phone data")
    # print(phone_data.shape)
    # print(phone_label.shape)

    # dataset_path = os.path.join(os.getcwd(), "wisdm/raw/watch")   
    # watch_data, watch_label = getGroup(dataset_path, "accel")
    # print("Watch data")
    # print(watch_data.shape)
    # print(watch_label.shape)
