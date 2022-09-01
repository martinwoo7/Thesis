from WiSARD.WCDS import test, wcds, clusterers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import time
import math

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale, Normalizer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from evolving.util import load_dataset

# from fishdbc_clustering import distance

# Function to return specific index for chunks of data
def getIndex(index, train_size, window_size, limit):
    train_start = index

    if index == 0:
        train_end = index + train_size
    else:
        train_end = index + window_size

    test_start = train_end
    test_end = min(test_start + window_size, limit)

    return train_start, train_end, test_start, test_end

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
    data, labels = datasets.make_blobs(n_samples=1000, centers=5)
    # beta = 18 for blobs or 20?
elif x == 5:
    print("Loading Iris")
    data, labels = datasets.load_iris(return_X_y=True)
    # beta = 10 for iris

# Parameters
# OMEGA = 250 # Determines when a discriminator expires?
# DELTA = 50
# GAMMA = 50 # Encoding resolution
# BETA = 21 # Increasing this also seems to make the alg run slower; length of addresses
EPSILON = 0.1
MU = 1
DIM = 9 # Is this talking about how many features there are?
for o in [250]:
    for a in [25]:
        for bet in [30]:
            print("Start clustering with Omega={}, Delta={}, Gamma={}, Beta={}, Epsilon={}, Mu={}".format(o, a, a, bet, EPSILON, MU))
            c_online = wcds.WCDS(
                omega=o,
                delta=a,
                gamma=a,
                epsilon=EPSILON,
                dimension=DIM,
                beta=bet,
                mu=MU
            )

            # 
            data = scale(data)
            minmaxscaler = MinMaxScaler()
            minmaxscaler.fit(data)
            data = minmaxscaler.transform(data)

            fig, axs = plt.subplots(2)
            fig.set_figheight(5)
            fig.set_figwidth(5)
            fig.suptitle('Results of Clustering')
            axs[0].set(xlabel="Window", ylabel="ARI")
            axs[1].set(xlabel="Window", ylabel="NMI")

            for ax in axs.flat:
                ax.label_outer()

            for ax in axs.flat:
                ax.set_ylim(0,1)

            time_count = 0
            returned_labs = []
            plotting_ari = []
            plotting_nmi = []
            plot_time = []
            for observation in data:
                # k is the id of the discriminator that absorbed the observation
                k, _ = c_online.record(observation, time_count)
                returned_labs.append(k)

                # finished = round(100 * (time_count / len(data)))
                # if finished in [25, 50, 75, 100]:
                #     print("Finished processing ", int(finished), "% of ", len(data), " events")

                labels_known = labels[:time_count + 1]
                ari = adjusted_rand_score(returned_labs, labels_known)
                nmi = normalized_mutual_info_score(returned_labs, labels_known)

                if time_count % 1000 == 0:
                    plotting_ari.append(ari)
                    plotting_nmi.append(nmi)
                    plot_time.append(time_count)
                    axs[0].plot(plot_time, plotting_ari, 'o-', color="blue")
                    axs[1].plot(plot_time, plotting_nmi, 'o-', color="red")
                time_count += 1
            print("Done processing")
            plt.show()

            ari = adjusted_rand_score(returned_labs, labels_known)
            nmi = normalized_mutual_info_score(returned_labs, labels_known)
            print("ARI is: ", ari)
            print("NMI is: ", nmi)

            print("Begin Offline")
            n_clusters = None
            threshold = 1
            c_offline = clusterers.MergeClustering(n_clusters=n_clusters, distance_threshold=threshold)
            actual_clusters = c_offline.fit(c_online.discriminators)
            print("Formed {} Clusters.".format(len(np.unique(list(actual_clusters.values())))))






