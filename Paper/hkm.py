import numpy as np
import math
from numpy.lib.arraysetops import unique
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import NoMatch
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, scale, LabelEncoder
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn import datasets

from evolving.util import load_dataset
from copy import deepcopy
from itertools import compress


class km():
    def __init__(self, data, n_clusters=8):
        self.k = n_clusters
        self.features = len(data[0])
        self.prototypes = self.random_prototype(self.k, self.features, data)
        self.old_proto = np.empty(self.prototypes.shape)
        self.old_E = 0

    def train(self, training_data, max_iteration=300):
        count = 0
        finish = False

        while (count < max_iteration) and (not finish):
            self.clusters = [[] for i in range(self.k)]
            self.cluster_labels = np.zeros(len(training_data))

            for i, input in enumerate(training_data):
                prototype_winner_i = self.calculate_winner(input, self.prototypes)
                self.clusters[prototype_winner_i].append(input) # To list?
                self.cluster_labels[i] = prototype_winner_i

            for i, cluster in enumerate(self.clusters):
                if cluster == []:
                    continue
                else:
                    zipped = zip(*cluster)
                    self.prototypes[i] = [item2 / len(cluster) for item2 in [sum(item) for item in zipped]]
            error = 0
            for j,prototype in enumerate(self.prototypes):
                temp = 0
                for item in self.clusters[j]:
                    temp += np.linalg.norm(item - prototype) ** 2
                error = error + temp

            if self.old_E == error:
                print("Minimum Error reached")
                finish = True
            if np.array_equal(self.prototypes, self.old_proto):
                print("Clusters stabilized")
                finish = True

            self.old_E = error
            self.old = deepcopy(self.prototypes)

            if (count + 1) >= max_iteration:
                print("Max epoch reached")
            count += 1
    
    def fit(self, x):
        labels = np.zeros(len(x))
        for i, input in enumerate(x):
            prototype_winner_i = self.calculate_winner(input, self.prototypes)
            labels[i] = prototype_winner_i
        return labels

    def calculate_winner(input, prototypes):
        for i, prototype in enumerate(prototypes):
            dist = np.linalg.norm(input - prototype)

            if i == 0:
                prev = dist
                answer = i
            else:
                if dist < prev:
                    prev = dist
                    answer = i
        return answer

    def random_prototype(k, features, data):
        return np.asarray([(data[index]) for index in [i for i in np.random.randint(0, len(data), k)]])

def _input(message, input_type=str):
    while True:
        try:
            a = input_type(input(message))
            if a > 5 or a < 1:
                raise ValueError
            return a
        except ValueError:
            print("Provide an acceptable input")

x = _input("Which MobiAct dataset to load? (1) Full dataset, (2) Partial dataset, (3) Mobi scenario, (4) blobs, (5) iris:", int)

if x == 1:
    print("Trying to load complete MobiAct data")
    try:
        data = np.load('mobiactdata.npy', allow_pickle=True)
        labels = np.load('mobiactlabels.npy', allow_pickle=True)
    except FileNotFoundError:
        print("No mobiact data saved! Starting data reading from dataset")
        data, labels = load_dataset.load_dataset("mobicomplete")
        print("Saving to np files")
        np.save('mobiactdata.npy', data)
        np.save('mobiactlabels.npy', labels)
    else:
        print("Successfully loaded full mobiacts data")
elif x == 2:
    print("Trying to load partial MobiAct dataset")
    try:
        data = np.load('mobiactpartialdata.npy', allow_pickle=True)
        labels = np.load('mobiactpartiallabels.npy', allow_pickle=True)
    except FileNotFoundError:
        print("No mobiact data saved! Starting data reading from dataset")
        data, labels = load_dataset.load_dataset("mobipartial")
        print("Saving to np files")
        np.save('mobiactpartialdata.npy', data)
        np.save('mobiactpartiallabels.npy', labels)
    else:
        print("Successfully loaded partial mobiacts data")
elif x == 3:
    print("Loading mobiact scenario")
    data, labels = load_dataset.load_dataset("mobiscenario")
elif x == 4:
    print("Loading Blobs")
    data, labels = datasets.make_blobs(n_samples=500, centers=3)
    # beta = 18 for blobs or 20?
elif x == 5:
    print("Loading Iris")
    data, labels = datasets.load_iris(return_X_y=True)
    # beta = 10 for iris

data = scale(data)
minmaxscaler = Normalizer()
minmaxscaler.fit(data)
data = minmaxscaler.transform(data)

def getIndex(index, train_size, window_size, limit):
    train_start = index

    if index == 0:
        train_end = index + train_size
    else:
        train_end = index + window_size

    test_start = train_end
    test_end = min(test_start + window_size, limit)

    return train_start, train_end, test_start, test_end

le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)

# layer1_labels = ["MVG" if (x == "WAL" or x == "STN" or x == "CSI" or x == "CSO") else "NMV" for x in labels]
# layer1_labels = ["SIT" if (x == "SIT" or x == "CSO" or x == "CSI") else "NSI" for x in labels]
# Parameters

print("Total number of classes returned")
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# Trying to cluster moving vs non-moving
# Moving includes WAL, CSO, CSI, and STN
# Non-moving incldues STD, SIT

# starting_centers = []
# first = True
# for act in unique_elements:
#     k = 1
#     # print("Starting sorting for ", act)
#     act_index = (labels == act)
#     temp_labels = labels[act_index]
#     temp_data = list(compress(data, act_index))

#     smolkm = KMeans(init="k-means++", n_clusters=k).fit(temp_data)
#     centers = smolkm.cluster_centers_
#     # print(centers)
#     if first:
#         starting_centers = centers
#         first = False
#     else:
#         # print("Centers: ",centers)
#         # print("ARI is: ", adjusted_rand_score(smolkm.labels_, temp_labels))
#         starting_centers = np.vstack((starting_centers, centers))

# print("Potential centers: ", starting_centers)

K = [6]

for k in K:
    # km = KMeans(init="k-means++", n_clusters=K, verbose=0)
    km = KMeans(init="k-means++", n_clusters=k, verbose=0)
    print("Begin clustering with k={}".format(k))
    km.fit(data)
    return_labels = km.labels_
    print("ARI is ", adjusted_rand_score(return_labels,labels))
    print("NMI is ", normalized_mutual_info_score(return_labels, labels))
    print("Accuracy is ", accuracy_score(return_labels, labels))
    print("\n")

# limit = len(data)
# index = 0
# test_end = 0
# train_size = 1000
# window_size = 1000

# plotting_index = []
# plotting_ari = []
# plotting_nmi = []

# fig, axs = plt.subplots(2)
# fig.set_figheight(5)
# fig.set_figwidth(5)
# fig.suptitle('Results of Clustering')

# axs[0].set(xlabel="Window", ylabel="ARI")
# axs[1].set(xlabel="Window", ylabel="NMI")

# for ax in axs.flat:
#     ax.label_outer()

# for ax in axs.flat:
#     ax.set_ylim(0,1)

# while test_end < limit:
#     train_start, train_end, test_start, test_end = getIndex(index, train_size, window_size, limit)
#     train_data = data[train_start:train_end]
#     index = test_start

#     plotting_index.append(train_end)

#     km.fit(train_data)
#     return_labels = km.labels_
#     nknown = len(return_labels)
#     labels_known = labels[:nknown]
    
#     ari = adjusted_rand_score(labels_known, return_labels)
#     nmi = normalized_mutual_info_score(return_labels, labels_known)
#     plotting_ari.append(ari)
#     plotting_nmi.append(nmi)
#     axs[0].plot(plotting_index, plotting_ari, 'o-', color="blue")
#     axs[1].plot(plotting_index, plotting_nmi, 'o-', color="red")
#     next_center = km.cluster_centers_
#     km = KMeans(init=next_center, n_clusters=K, verbose=0)
    # assert False
# km.fit(data)
# return_labels = km.labels_


# le = LabelEncoder()
# le.fit(layer1_labels)
# layer1_labels = le.transform(layer1_labels)

# print("Total number of classes clustered")
# unique_elements, counts_elements = np.unique(return_labels, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))

# print("Centers: ", km.cluster_centers_)