#%%
import os
os.environ["OMP_NUM_THREADS"] = "4"
# import sys
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tmap as tm

from scipy.spatial import distance as dst
from faerun import Faerun
# from dtw import * 

from evolving.util import load_dataset
from sklearn import model_selection
from sklearn import preprocessing as preproc
from sklearn.utils import class_weight
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, homogeneity_score, completeness_score, v_measure_score
from tensorflow.keras import (callbacks, datasets, layers, models, optimizers,
                              utils, preprocessing, regularizers)
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from imblearn.over_sampling import SMOTE
from flexible_clustering import FISHDBC
from dtw import *

from operator import itemgetter
from itertools import groupby

from visual import Visual
from te_utils import *
#%%

def _input(message, input_type=str):
    while True:
        try:
            a = input_type(input(message))
            if a not in range(1, 7):
                raise ValueError
            return a
        except ValueError:
            print("Provide an acceptable input")


def distance(x, y):
    return dst.euclidean(x, y)
    # return dst.minkowski(x, y, 1)
    # return dst.cityblock(x, y)
    # return dtw(x, y, distance_only=True).distance


EF = 50
MIN_SAMPLES = 1
MIN_CLUSTER_SIZE = 10
CLUSTER_SELECTION_EPSILON = 0

def cluster(model, mode=1, data=None, labels=None, gen=False, layer=0):
    '''
    Takes in a CNN_LSTM model
    '''
    # cloned_model = models.clone_model(model)
    # cloned_model.set_weights(model.get_weights())
    #  I want to remove the last two layers (Dense, Dense)
    if layer == 0:
        print('Pass in proper layer number')
        assert False

    popped_model = models.Model(inputs=model.input, outputs=model.layers[-layer].output)
    # popped_model.summary()

    if mode == 1:
        print("Clustering using FISHDBC")
        
        if not gen:
            predict_labels = model.predict(data)
            predict_labels = [y.argmax() for y in predict_labels]
            true_labels = [y.argmax() for y in labels]

            differences = [predict_labels[i] == true_labels[i] for i in range(len(true_labels))]
            difference_i = [i for i, value in enumerate(differences) if not value]

            data = [np.delete(data, difference_i, axis=0)]
            true_labels = [true_labels[i] for i, value in enumerate(differences) if value]

            output = popped_model.predict(data)
            # num_clusters = len(np.unique(correct_labels))
        else:
            true_labels = []
            # test_labels = []
            outputs = None
            
            gen = train_generator(None)
            for _ in range(15):
                # cluster the dataset one activity batch at a time
                data, labels = next(gen)
                # I want to only pass in correct classifications
                test_outputs = model.predict(data)
                test_outputs = [y.argmax() for y in test_outputs] 
                test_labels = [y.argmax() for y in labels]

                differences = [test_outputs[i]==test_labels[i] for i in range(len(test_labels))]
                difference_i = [i for i, value in enumerate(differences) if not value]

                data = [np.delete(data[0], difference_i, axis=0), np.delete(data[1], difference_i, axis=0), np.delete(data[2], difference_i, axis=0)]
                test_labels  = [test_labels[i] for i, value in enumerate(differences) if value]

                cnn_output = popped_model.predict(data)

                if outputs is None:
                    outputs = cnn_output
                else:
                    outputs = np.vstack((outputs, cnn_output))

                true_labels += test_labels

        print(82 * "_")
        print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette") 
        t0 = time.time()

        fishdbc = FISHDBC(distance, ef=EF, min_samples=MIN_SAMPLES)
        fishdbc.update(output)
        labs, _, _, ctree, _, _ = fishdbc.cluster(min_cluster_size=MIN_CLUSTER_SIZE, cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON)

        fit_time = time.time() - t0
        results = ["FISHDBC", fit_time, 0]

        clustering_metrics = [
            homogeneity_score,
            completeness_score,
            v_measure_score,
            adjusted_rand_score,
            normalized_mutual_info_score,
        ]
        results += [m(labs, true_labels) for m in clustering_metrics]
        results += [
        0
        ]
        formatter_result = (
            "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
        )
        print(formatter_result.format(*results))
        values, counts = np.unique(labs, return_counts=True)
        print(values)
        print(counts)
        print(82 * "_")

        return output, true_labels, labs
        
    
    elif mode == 2:

        print("Kmeans clustering")

        if not gen:
            predict_labels = model.predict(data)
            predict_labels = [y.argmax() for y in predict_labels]
            true_labels = [y.argmax() for y in labels]

            differences = [predict_labels[i] == true_labels[i] for i in range(len(true_labels))]
            difference_i = [i for i, value in enumerate(differences) if not value]

            if len(data)==3:
                data = [np.delete(data[0], difference_i, axis=0), np.delete(data[1], difference_i, axis=0), np.delete(data[2], difference_i, axis=0)]
            else:
                data = [np.delete(data, difference_i, axis=0)]
            correct_labels = [true_labels[i] for i, value in enumerate(differences) if value]

            output = popped_model.predict(data)
            num_clusters = len(np.unique(correct_labels))

        else:

            true_labels = []
            true_data = []

            gen = train_generator(None)
            for i in range(15):
                data, labels = next(gen)

                first_outputs = model.predict(data)
                first_outputs = [y.argmax() for y in first_outputs]
                test_labels = [y.argmax() for y in labels]

                differences = [first_outputs[i]==test_labels[i] for i in range(len(test_labels))]
                difference_i = [i for i, value in enumerate(differences) if not value]

                data = [np.delete(data[0], difference_i, axis=0), np.delete(data[1], difference_i, axis=0), np.delete(data[2], difference_i, axis=0)]

                correct_labels  = [test_labels[i] for i, value in enumerate(differences) if value]
                true_labels += correct_labels
                cnn_output = popped_model.predict(data) # cnn_output is a np array
                
                if i == 0:
                    true_data = cnn_output
                else:
                    true_data = np.vstack((true_data, cnn_output))
                
                # true_data += cnn_output
        print("final data shape is: ", output.shape)
        print("Number of clusters: ", num_clusters)
        print(82 * "_")
        print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette") 
        kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=4)
        cluster_labels = bench_k_means(kmeans=kmeans, name="k-means++", data=output, labels=correct_labels)
        print(82 * "_")
        return output, correct_labels, cluster_labels
    else:
        print("Mode not found. Clustering skipped")

x = _input("(1) Manual engineering. (2) Stacked TE. (3) Timeseries w/ pad. (4) multi-head TE. (5) testing model", int)

if x == 1:
    # training_data = getDataset("stacked_training_data.hdf5", "stacked_training_dataset")
    # training_labels = getDataset("stacked_training_labels.hdf5", "stacked_training_labels")
    data, labels = load_dataset.load_dataset("mobiscenario")
    le = preproc.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    
    training_data, training_labels = fuse(data, labels, mode=1)
    scaler = preproc.MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    print(training_data.shape)
    # training_labels = np.argmax(training_labels, axis=1)
    training_labels = training_labels.reshape(-1)
    print(training_labels.shape)

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette")

    kmeans = KMeans(init="k-means++", n_clusters=6, random_state=42, n_init=4)
    bench_k_means(kmeans=kmeans, name="k-means++", data=training_data, labels=training_labels)
    
    pca = PCA(n_components=6).fit(training_data)
    kmeans = KMeans(init=pca.components_, n_clusters=6, n_init=1)
    bench_k_means(kmeans=kmeans, name="PCA-based", data=training_data, labels=training_labels)
    
    print(82 * "_")
    print("skipped")

elif x == 2:
    # Base TE
    # stacked dataset

    # This is just used to create the normalizer
    training_data, training_labels = load_train("stacked")
    testing_data, testing_labels = load_test("stacked")
    print(training_labels.shape)

    scaler = preproc.MinMaxScaler()
    scaler = scaler.fit(training_data)
    training_data = scaler.transform(training_data)
    training_data = training_data.reshape(-1, 1, 82)
    training_labels = training_labels.reshape(-1, 1, 16)
    model_name = 'models/cnn_LSTM_Mar_2022_test'
    # model_name = "models/best_CNN"

    try:
        cnn_lstm = models.load_model(model_name)
        cnn_lstm.summary()
         
    except:
        print("beginning creation and training - model not found")

        cnn_lstm = models.Sequential()
        cnn_lstm.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=(None, 82)))
        cnn_lstm.add(layers.LSTM(100, return_sequences=True))
        cnn_lstm.add(layers.Dense(16, activation='softmax'))
        cnn_lstm.summary()
        # optimizing = optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
        optimizing = optimizers.Adam(learning_rate=0.001)
        cnn_lstm.compile(loss='categorical_crossentropy', optimizer=optimizing, metrics=['accuracy'])

        # checkpoint = callbacks.ModelCheckpoint(
        #     "models/best_CNN_5",
        #     monitor="val_accuracy",
        #     verbose=0,
        #     save_best_only=True,
        #     save_weights_only=False,
        #     mode="auto"
        # )
        history = cnn_lstm.fit(training_data, training_labels, epochs=64, validation_split=0.2, verbose=1, batch_size=128)

        # cnn_lstm.save(model_name, save_format='h5')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

    testing_data = scaler.transform(testing_data)
    testing_data = testing_data.reshape(-1, 1, 82)
    testing_labels = testing_labels.reshape(-1, 1, 16)
    _, test_acc = cnn_lstm.evaluate(testing_data, testing_labels, batch_size=128)
    print("Accuracy of final model is:", test_acc)
    # print("Beginning clustering into FISHDBC and kmeans")

    # best_model = models.load_model('models/best_CNN_5')
    # _, test_acc = best_model.evaluate(x_test, y_test, verbose=1)
    # print("Accuracy of best model is:", test_acc)

elif x == 3:
    # TimeDistributed TE aka time slice
    #     
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, test_size=0.25, random_state=36)
    DATASET = "raw_400"
    training_data, training_labels = load_train(DATASET)
    testing_data, testing_labels = load_test(DATASET)
    print(training_data.shape, training_labels.shape)
    print(testing_data.shape, testing_labels.shape)

    orig_shape = training_data.shape
    scaler = preproc.Normalizer().fit(training_data.reshape(-1, orig_shape[2]))
    training_data = scaler.transform(training_data.reshape(-1, orig_shape[2])).reshape(orig_shape)

    model_name = "models/timedis/mobi_512"
    n_features, n_steps = orig_shape[-1], 4
    n_timesteps = orig_shape[1] // n_steps
    training_data = training_data.reshape(orig_shape[0], n_steps, n_timesteps, n_features)
    # n_features, n_length = 82, 16
    print("New shape is ", training_data.shape)
    
    epochs, batch = 64, 128

    try:
        print("Looking for time slice model")
        cnn_lstm = models.load_model(model_name)
        cnn_lstm.summary()
        assert False
    except:
        print("Model not found, beginning model creating and training")
        cnn_lstm = models.Sequential()
        # cnn_lstm.add(layers.TimeDistributed(layers.Masking(), input_shape=(None, n_steps, n_time, n_features)))
        cnn_lstm.add(layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=5, activation='relu'), input_shape=(n_steps, n_timesteps, n_features)))
        cnn_lstm.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
        cnn_lstm.add(layers.TimeDistributed(layers.Dropout(0.5)))
        cnn_lstm.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        cnn_lstm.add(layers.TimeDistributed(layers.Flatten()))
        # cnn_lstm.add(layers.TimeDistributed(layers.Masking(mask_value=0)))
        cnn_lstm.add(layers.Bidirectional(layers.LSTM(64, recurrent_dropout=0.5, return_sequences=True)))
        cnn_lstm.add(layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.5)))
        cnn_lstm.add(layers.Dense(128, activation='relu'))
        cnn_lstm.add(layers.BatchNormalization())
        cnn_lstm.add(layers.Dense(16, activation='softmax'))
        # cnn_lstm.summary()

        optimizing = optimizers.Adam(learning_rate=0.001)
        cnn_lstm.compile(loss='categorical_crossentropy', optimizer=optimizing, metrics=['accuracy'])

        history = cnn_lstm.fit(training_data, training_labels, epochs=epochs, validation_split=0.2, verbose=1, shuffle=True, batch_size=batch)
        # history = cnn_lstm.fit(train_generator(singular=True), epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=15, validation_data=validate_generator(42, singular=True), validation_steps=15)
        # cnn_lstm.save(model_name, save_format='h5')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
    else:
        print("Model found and loaded!")

    orig_test_shape = testing_data.shape
    testing_data = scaler.transform(testing_data.reshape(-1, orig_test_shape[2])).reshape(orig_test_shape)
    testing_data = testing_data.reshape(-1, n_steps, n_timesteps, n_features)
    _, test_acc = cnn_lstm.evaluate(testing_data, testing_labels, batch_size=batch, verbose=0)
    print("Accuracy is:", test_acc)
    
    reduced_data, orig_labels, new_labels = cluster(cnn_lstm, mode=2, data=testing_data, labels=testing_labels, layer=4)
    visual = Visual(np.array(reduced_data), np.array(orig_labels))
    # visual.densmap(neighbours=300, dist=0.01)
    visual.tmap(100, 10)
    # visual.clusterplot()
    
    

elif x == 4:
    # Mutli head TE
    # segmented - many to one w/ extra repeats to reach 16
    # data, labels = fuse()
    # scaler = preproc.StandardScaler()
    # scaler = scaler.fit(data.reshape(-1, data.shape[-1]))
    DATASET = "SLH"
    training_data, training_labels = load_train(DATASET)
    testing_data, testing_labels = load_test(DATASET)
    print(training_data.shape, training_labels.shape)
    print(testing_data.shape, testing_labels.shape)

    training_data = np.concatenate((training_data, testing_data))
    training_labels = np.concatenate((training_labels, testing_labels))

    orig_shape = training_data.shape
    scaler = preproc.Normalizer().fit(training_data.reshape(-1, orig_shape[2]))
    training_data = scaler.transform(training_data.reshape(-1, orig_shape[2])).reshape(orig_shape)

    model_name = "models/bidirect/SLH"
    n_features, n_steps = orig_shape[-1], 4
    n_timesteps = orig_shape[1] // n_steps
    training_data = training_data.reshape(-1, n_steps, n_timesteps, n_features)
    print("New shape is ", training_data.shape)
    epochs, batch = 64, 64

    try:
        print("Looking for multi-headed model")
        model = models.load_model(model_name)
        # cnn_lstm.summary()
    except:
        inputs1 = layers.Input(shape=(None, n_timesteps, n_features))
        conv1 = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding="same"))(inputs1)
        conv12 = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))(conv1)
        drop1 = layers.TimeDistributed(layers.Dropout(0.5))(conv12)
        pool1 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(drop1)
        flat1 = layers.TimeDistributed(layers.Flatten())(pool1)
        
        # Head 2
        inputs2 = layers.Input(shape=(None, n_timesteps, n_features))
        conv2 = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding="same"))(inputs2)
        conv22 = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=7, activation='relu'))(conv2)
        drop2 = layers.TimeDistributed(layers.Dropout(0.5))(conv22)
        pool2 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(drop2)
        flat2 = layers.TimeDistributed(layers.Flatten())(pool2)

        # Head 3
        inputs3 = layers.Input(shape=(None, n_timesteps, n_features))
        conv3 = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=11, activation='relu', padding="same"))(inputs3)
        conv32 = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=11, activation='relu'))(conv3)
        drop3 = layers.TimeDistributed(layers.Dropout(0.5))(conv32)
        pool3 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(drop3)
        flat3 = layers.TimeDistributed(layers.Flatten())(pool3)

        merged = layers.Concatenate()([flat1, flat2, flat3])
        bilstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True, recurrent_dropout=0.5))(merged)
        bilstm2 = layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.5))(bilstm1)

        dense1 = layers.Dense(128, activation='relu')(bilstm2)
        batchnorm = layers.BatchNormalization()(dense1)
        outputs = layers.Dense(16, activation='softmax')(batchnorm)

        model = models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # plot_model(model, show_shapes=True, to_file="multichannel.png")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # history = model.fit(train_generator(scaler), epochs=epochs, verbose=1, shuffle=True, steps_per_epoch=15, validation_data=validate_generator(42), validation_steps=15)

        history = model.fit([training_data, training_data, training_data], training_labels, epochs=epochs, verbose=1, shuffle=True, validation_split=0.3, batch_size=batch)

        model.save(model_name, save_format='h5')
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

    # _, test_acc = model.evaluate(validate_generator(42), verbose=0, steps=15)

    # orig_test_shape = testing_data.shape
    # testing_data = scaler.transform(testing_data.reshape(-1, orig_test_shape[2])).reshape(orig_test_shape)
    # testing_data = testing_data.reshape(-1, n_steps, n_timesteps, n_features)
    # _, test_acc = model.evaluate([testing_data, testing_data, testing_data], testing_labels, verbose=0, batch_size=batch)
    # print("Accuracy is:", test_acc)
    print("Begin clustering and all that fun stuff")

    reduced_data, orig_labels, new_labels = cluster(model, mode=2, data=[training_data, training_data, training_data], labels=training_labels, layer=4)
    print(np.array(reduced_data).shape, np.array(orig_labels).shape)
    visual = Visual(np.array(reduced_data), np.array(orig_labels))
    # visual.densmap(neighbours=300, dist=0.01)
    visual.tmap(100, 10)
    # visual.clusterplot()

elif x == 5:
    # improve TE model
    # segmented 2 - many to many with overlap
    # data, labels = fuse()
    # scaler = preproc.StandardScaler()
    # scaler = scaler.fit(data.reshape(-1, data.shape[-1]))

    # pass scaler in again
    # state = random.randint(10, 100)

    model_name = 'models/improve_TE/improve_test'
    training_data, training_labels = load_train("raw")
    testing_data, testing_labels = load_test("raw")
    print(training_data.shape, training_labels.shape)

    scaler = preproc.MinMaxScaler()
    orig_shape = training_data.shape
    scaler = scaler.fit(training_data.reshape(-1, orig_shape[2]))
    training_data = scaler.transform(training_data.reshape(-1, orig_shape[2])).reshape(orig_shape)

    n_features = orig_shape[2]
    # training_data = training_data.reshape(-1, 1, 82)
    # training_labels = training_labels.reshape(-1, 16)

    try:
        cnn_lstm = models.load_model(model_name)
        cnn_lstm.summary()
         
    except:
        print("beginning creation and training - model not found")

        cnn_lstm = models.Sequential()
        cnn_lstm.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(None, n_features)))
        cnn_lstm.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        cnn_lstm.add(layers.LSTM(128, return_sequences=True))
        cnn_lstm.add(layers.LSTM(64, return_sequences=False))
        cnn_lstm.add(layers.Dense(128, activation='relu'))
        cnn_lstm.add(layers.Dropout(0.5))
        cnn_lstm.add(layers.Dense(16, activation='softmax'))
        cnn_lstm.summary()
        optimizing = optimizers.Adam(learning_rate=0.001)
        cnn_lstm.compile(loss='categorical_crossentropy', optimizer=optimizing, metrics=['accuracy'])

        # remmeber to do callbacks
        # history = cnn_lstm.fit(train_generator2(state), steps_per_epoch=15, epochs=64, verbose=1, shuffle=True, validation_steps=15, validation_data=validate_generator2(state))
        history = cnn_lstm.fit(training_data, training_labels, epochs=64, batch_size=64, verbose=1, shuffle=True, validation_split=0.2)
        # cnn_lstm.save(model_name, save_format='h5')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

    else:
        print("model found and loaded")

    test_shape = testing_data.shape
    testing_data = scaler.transform(testing_data.reshape(-1, test_shape[2])).reshape(test_shape)
    # testing_data = testing_data.reshape(-1, 1, 82)
    # testing_labels = testing_labels.reshape(-1,16)
    # _, test_acc = cnn_lstm.evaluate(validate_generator2(state), verbose=1, steps=15)
    _, test_acc =  cnn_lstm.evaluate(testing_data, testing_labels, batch_size=64, verbose=0)
    print("Accuracy of final model is:", test_acc)
    # print("Beginning clustering into FISHDBC and kmeans")
