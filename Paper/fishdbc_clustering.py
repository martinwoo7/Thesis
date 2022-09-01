import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import hdbscan
# import math
# import time
import collections
# import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, LabelEncoder, scale
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.spatial import distance as dst

from flexible_clustering import FISHDBC
from evolving.util import load_dataset
from IPython.display import clear_output

# This is the distance function the FISHDBC needs to work. Can be any arbitrary distance
def distance(x, y):
    return dst.euclidean(x, y)

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
            if a > 4 or a < 1:
                raise ValueError
            return a
        except ValueError:
            print("Provide an acceptable input")

x = _input("Which MobiAct dataset to load? (1) Full dataset, (2) Partial dataset, (3) Mobi scenario, (4) blobs:", int)

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

# x, y = data[:,0], data[:,1]
min_cluster_size = [100]
# min_cluster somewhere in 500+ range but <800
# cluster_epsilon = [7]
cluster_epsilon = [0.1]
# cluster_epsilon somewhere between 5 and 10 is ideal or maybe a bit more than 10 but < 20
for min_clus in min_cluster_size:
    for clus in cluster_epsilon:
        # Parameters for FISHDBC
        EF = 50
        MIN_SAMPLES = 1
        MIN_CLUSTER_SIZE = min_clus
        CLUSTER_SELECTION_EPSILON = clus
        DISTANCE = "euclidean"

        # Paramters for main clustering function
        nsamples = len(labels) # How many points to cluster
        test_end = 0 # The end index for each iteration
        index = 0 # The starting index for each iteration
        limit = nsamples # Determines when to stop clustering
        train_size = 500 # Initial datapoint fed in to create a baseline
        window_size = 500 # Window size


        data = data[:nsamples]
        labels = labels[:nsamples]

        # Change string labels to numerical
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)

        # TODO: Consider scaling everything and saving that scale to apply onto new data
        # Test out the results before and after scaling then decide
        # Minmax, Normalize, Standardize
        # Scaling the data actally led to worse performance. Alot more points were counted as noise
        # and only something like 50% of all data was clustered

        # TODO: I want to try feeding in the whole dataset and use those as cluster centers
        # to cluster the new incoming points ~~ DONE

        data = scale(data)
        minmaxscaler = Normalizer()
        minmaxscaler.fit(data)
        data = minmaxscaler.transform(data)

        fishdbc = FISHDBC(distance, ef=EF, min_samples=MIN_SAMPLES)
        # fishdbc = FISHDBC(distance)



        print("Loaded " + str(len(data)) + " pieces of data!")
        print("Starting Clustering :)")
        print("## Parameters are: EF={}, min_samples={},"
            "min_clusters={}, length={}, distance measure={}, cluster_epsilon={}".format(EF,MIN_SAMPLES,MIN_CLUSTER_SIZE,window_size,DISTANCE,CLUSTER_SELECTION_EPSILON))

        plotting_index = []
        plotting_ari = []
        plotting_nmi = []

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

        while test_end < limit:
            train_start, train_end, test_start, test_end = getIndex(index, train_size, window_size, limit)
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]
            index = test_start
            plotting_index.append(train_end)
            finished = round(100 * (train_end / limit))
            if finished in [25, 50, 75, 100]:
                print("Finished processing ", int(finished), "% of ", limit, " events")


            fishdbc.update(train_data)
            nknown = len(fishdbc.data)
            labs, probs, stabilities, ctree, slt, mst = fishdbc.cluster(min_cluster_size=MIN_CLUSTER_SIZE, cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON)
            
            # clusters = collections.defaultdict(set)

            # for parent, child, lambda_val, child_size in ctree[::-1]:
            #     if child_size == 1:
            #         clusters[parent].add(child)
            #     else:
            #         assert len(clusters[child]) == child_size
            #         clusters[parent].update(clusters[child])
            # clusters = sorted(clusters.items())
            xknown, yknown, labels_known = data[:nknown], labels[:nknown], labels[:nknown]
            # color = ['rgbcmyk'[l & 7] for l in labels_known]

            # plt.figure(figsize=(9, 9))
            # plt.gca().set_aspect('equal')
            # print(len(xknown), len(yknown))
            # plt.scatter(xknown, yknown, c=color, linewidth=0)
            # plt.show(block=False)
            # for _, cluster in clusters:
            #     plt.gca().clear()
            #     color = ['kr'[i in cluster] for i in range(nknown)]
            #     plt.scatter(xknown, yknown, c=color, linewidth=0)
            #     plt.draw()

            ari = adjusted_rand_score(labels_known, labs)
            nmi = normalized_mutual_info_score(labs, labels_known)

            # plotting_ari.append(ari)
            # plotting_nmi.append(nmi)
            # # ax[0].scatter(xknown, yknown)
            # axs[0].plot(plotting_index, plotting_ari, 'o-', color="blue")
            # axs[1].plot(plotting_index, plotting_nmi, 'o-', color="red")

        print("Done processing")
        # plt.show()

        clustered = (labs >= 0)
        print("Final ARI is: ", adjusted_rand_score(labs[clustered], labels_known[clustered]))
        print("Final NMI is: ", normalized_mutual_info_score(labs[clustered], labels_known[clustered]))

        print("Total percentage of data clustered: ", np.sum(clustered) / nsamples)
        print("\n")


