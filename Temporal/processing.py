import os
import pandas as pd
import numpy as np
import h5py
import math

from tensorflow.keras import utils
from sklearn import model_selection
from collections import Counter
from scipy.stats import kurtosis, skew

DATASET_PATH = os.path.join(os.getcwd(), "data/Annotated Data/activities")
categorical = {"STD": 0, "WAL": 1, "JOG": 2, "JUM": 3, "STU": 4, "STN": 5, "SCH": 6, "SIT": 7, "CHU": 8, "CSI": 9, "CSO": 10, "LYI": 11, "FOL": 12, "FKL": 13, "BSC": 14, "SDL": 15}

data, labels = [], []

def fuse(step, data):
    reduced = []
    # reduced_labs = []

    reduced_data = list(zip(*[iter(data)] * step))

    for subset in reduced_data:
        # sublab = labs[i:i+step]

        subset = np.array(subset)
        row = []
        slope_array = []
        for j in range(data.shape[1]):
            # iterate through all columns
            # arr = [x[j] for x in subset]
            arr = subset[:, j]
            mean_array = np.mean(arr)
            median_array = np.median(arr)
            std_array = np.std(arr)
            kurt_array = kurtosis(arr)
            var_array = np.var(arr)
            skew_array = skew(arr)
            min_array = min(arr)
            max_array = max(arr)
            # hmean_array = hmean(arr)
            zrc_array = sum((arr[:-1] * arr[1:] < 0)) / arr.shape[0]
            total_array = [mean_array, median_array, std_array, kurt_array, var_array, skew_array, min_array, max_array, zrc_array]
            
            if len(row) == 0:
                row = total_array
                slope_array = [[min_array, max_array]]
            else:
                # row = np.append(row, total_array)
                row.extend(total_array)

                # slope_array = np.vstack((slope_array, [min_array, max_array]))
                slope_array.append([min_array, max_array])
        
        summ = 0
        # slope_array = np.array(slope_array)
        for axi in slope_array:
            summ += (axi[1] - axi[0])**2
        
        slope = math.sqrt(summ)
        row.extend([slope])

        reduced.append(row)

    return reduced

def grouper(n, iterable):
    return list(zip(*[iter(iterable)] * n))

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def gen(size, step, A):
    return [A[i : i + size] for i in range(0, len(A), step)]

# for folder in os.listdir(os.fsencode(DATASET_PATH)):
#     folderName = os.fsdecode(folder)
#     activity_path = DATASET_PATH + '/' + folderName
#     print("loading in %s data" % folderName)

#     for count, activity in enumerate(os.listdir(os.fsencode(activity_path))):
#         if activity == "WAL" or activity == "STD" or activity == "SIT":
#             if count % 3 != 0:
#                 continue

#         activity_file_path = os.path.join(activity_path + '/' + os.fsdecode(activity))
#         activity_file = pd.read_csv(activity_file_path)
        
#         activity_file = activity_file.drop(columns=["timestamp", "rel_time"])
#         activity_file = activity_file.to_numpy()

#         # n = 200
#         # for count, batch in enumerate(grouper(n, activity_file)):
#         #     batch = np.array(batch)
#         #     if batch.shape[0] == n:
#         #         label_list = batch[:, -1]
#         #         batch_subset = batch[:, :-1]

#         #         batch_subset = fuse(4, batch_subset)

#         #         labels.append(categorical[Most_Common(label_list)])
#         #         data.append(batch_subset)
#         n = 200
#         overlap = n
#         for count, batch in enumerate(gen(n, overlap, activity_file)):
#             batch = np.array(batch)
#             if batch.shape[0] == n:
#                 label_list = batch[:, -1]                                                                                                                                                                                                                                                                                                                                                                  
#                 batch_subset = batch[:, :-1]

#                 # batch_subset = fuse(2, batch_subset)

#                 labels.append(categorical[Most_Common(label_list)])
#                 data.append(batch_subset)


activity_file = pd.read_csv("SLH_1_1_annotated.csv")
activity_file = activity_file.drop(columns=["timestamp", "rel_time"])
activity_file = activity_file.to_numpy()
n = 200
overlap = n
for count, batch in enumerate(gen(n, overlap, activity_file)):
    batch = np.array(batch)
    if batch.shape[0] == n:
        label_list = batch[:, -1]                                                                                                                                                                                                                                                                                                                                                                  
        batch_subset = batch[:, :-1]

        # batch_subset = fuse(2, batch_subset)

        labels.append(categorical[Most_Common(label_list)])
        data.append(batch_subset)

data = np.array(data, dtype="float32")
changed_labels = utils.to_categorical(labels, num_classes=16)
print(data.shape, changed_labels.shape)

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, changed_labels, test_size=0.3)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

DATASET = "SLH"
# raw_overlap_manual_subject sorts training and testing based on subjects
# training will use first 80% of subjects and testing will use last 20%

# raw_manual_balance reads every 3rd for STD, SIT, and WAL to balance the numbers a bit
FILE_NAME = "datasets/" + DATASET + "/"
if not os.path.exists("datasets/" + DATASET):
    os.makedirs("datasets/" + DATASET)
    os.makedirs("datasets/" + DATASET + "/train")
    os.makedirs("datasets/" + DATASET + "/test")

f1 = h5py.File(FILE_NAME + "/train/" + DATASET + "_training_data.hdf5", "w")
f1.create_dataset(DATASET + "_training_data", data=x_train)
f1.close()

f2 = h5py.File(FILE_NAME + "/train/" + DATASET + "_training_labels.hdf5", "w")
f2.create_dataset(DATASET + "_training_labels", data=y_train)
f2.close()

f3 = h5py.File(FILE_NAME + "/test/" + DATASET + "_testing_data.hdf5", "w")
f3.create_dataset(DATASET + "_testing_data", data=x_test)
f3.close()

f4 = h5py.File(FILE_NAME + "/test/" + DATASET + "_testing_labels.hdf5", "w")
f4.create_dataset(DATASET + "_testing_labels", data=y_test)
f4.close()

   