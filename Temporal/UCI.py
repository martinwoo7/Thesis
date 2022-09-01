import math
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import h5py

import numpy as np
import pandas as pd
import tmap as tm
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial import distance as dst
from sklearn import preprocessing as preproc

from h5py._hl.files import File
from faerun import Faerun
from sklearn import metrics, model_selection
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from sklearn.metrics import adjusted_rand_score, mean_squared_log_error, normalized_mutual_info_score, accuracy_score, homogeneity_score, v_measure_score, completeness_score
from tensorflow.keras import (callbacks, datasets, layers, models, optimizers,
                              utils, preprocessing, regularizers)
from flexible_clustering import FISHDBC
from operator import itemgetter
from itertools import groupby
from visual import Visual
from uci_utils import *

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

def distance(x, y):
    return np.linalg.norm(x - y)
    # return dst.euclidean(x, y)
    # return dst.minkowski(x, y, 1)
    # return dst.cityblock(x, y)
    # return dtw(x, y, distance_only=True).distance

EF = 50
MIN_SAMPLES = 1
MIN_CLUSTER_SIZE = 20 # or 250 works
CLUSTER_SELECTION_EPSILON = 1
def cluster(model, mode=1, data=None, labels=None, layer=0):
    if layer == 0:
        print("wrong layer")
        assert False

    popped_model = models.Model(inputs=model.input, outputs=model.layers[-layer].output)

    if mode == 1:
        # EF = 50
        # MIN_SAMPLES = 1
        # MIN_CLUSTER_SIZE = 20 # or 250 works
        # CLUSTER_SELECTION_EPSILON = 1

        true_labels = []
        test_labels = []
        fishdbc = FISHDBC(distance, ef=EF, min_samples=MIN_SAMPLES)

        test_outputs = model.predict(data)
        test_outputs = [y.argmax() for y in test_outputs]
        test_labels = [y.argmax() for y in labels]

        differences = [test_outputs[i]==test_labels[i] for i in range(len(test_labels))]
        difference_i = [i for i, value in enumerate(differences) if not value]  

        data = np.delete(data, difference_i, axis=0)
        print(data.shape)

        test_labels  = [test_labels[i] for i, value in enumerate(differences) if value]

        cnn_output = popped_model.predict(data)
        true_labels += test_labels
        fishdbc.update(cnn_output)
        labs, _, _, ctree, _, _ = fishdbc.cluster(min_cluster_size=MIN_CLUSTER_SIZE, cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON)

        true_labels = np.array(true_labels)

        ari = adjusted_rand_score(labs, true_labels)
        nmi = normalized_mutual_info_score(labs, true_labels)
        print("ARI is: ", ari)
        print("NMI is: ", nmi)
        values, counts = np.unique(labs, return_counts=True)
        print(values, counts)

    elif mode == 2:
        predict_labels = model.predict(data)
        predict_labels = [y.argmax() for y in predict_labels]
        true_labels = [y.argmax() for y in labels]

        differences = [predict_labels[i]==true_labels[i] for i in range(len(true_labels))]
        difference_i = [i for i, value in enumerate(differences) if not value]  
        
        if len(data)==3:
            data = [np.delete(data[0], difference_i, axis=0), np.delete(data[1], difference_i, axis=0), np.delete(data[2], difference_i, axis=0)]
        else:
            data = np.delete(data, difference_i, axis=0)
        correct_labels  = [true_labels[i] for i, value in enumerate(differences) if value]

        output = popped_model.predict(data)
        num_clusters = len(np.unique(correct_labels))

        print("final data shape is: ", output.shape)
        print("Number of clusters: ", num_clusters)
        print(82 * "_")
        print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette") 
        kmeans = KMeans(init="k-means++", n_clusters=num_clusters, n_init=4)
        cluster_labels = bench_k_means(kmeans=kmeans, name="k-means++", data=output, labels=correct_labels)
        print(82 * "_")
        return output, correct_labels, cluster_labels


def _input(message, input_type=str):
    while True:
        try:
            a = input_type(input(message))
            if a not in range(0, 7):
                raise ValueError
            return a
        except ValueError:
            print("Provide an acceptable input")

x = _input("(0) Skip. (1) Tri. (2) ConvAE + LSTM. (3) LSTM AE. (4) Base TE. (5) TimeDist. (6) Improved TE. ", int)

if x == 0:
    print("Skipped")

if x == 1:
    print("Starting UCI HAR data processing")
    trainX, trainy, testX, testy = load_dataset()

    n_timestamps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    epochs, batch_size = 64, 128
    n_steps = 4
    n_timesteps = n_timestamps // n_steps
    model_name = "models/bidirect/uci_bidirect_test"

    trainX, testX = scale_data(trainX, testX, True)
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_timesteps, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_timesteps, n_features))
    try:
        print("Looking for UCI model")
        model = models.load_model(model_name)
        # uci_model.summary()

    except:
        print("Model not found: beginning creation of UCI model")
        # print(trainX.shape)

        # Head 1
        inputs1 = layers.Input(shape=(None, n_timesteps, n_features))
        conv1 = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))(inputs1)
        conv12 = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))(conv1)
        drop1 = layers.TimeDistributed(layers.Dropout(0.5))(conv12)
        pool1 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(drop1)
        flat1 = layers.TimeDistributed(layers.Flatten())(pool1)
        
        # Head 2
        inputs2 = layers.Input(shape=(None, n_timesteps, n_features))
        conv2 = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='same'))(inputs2)
        conv22 = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=7, activation='relu'))(conv2)
        drop2 = layers.TimeDistributed(layers.Dropout(0.5))(conv22)
        pool2 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(drop2)
        flat2 = layers.TimeDistributed(layers.Flatten())(pool2)

        # Head 3
        inputs3 = layers.Input(shape=(None, n_timesteps, n_features))
        conv3 = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=11, activation='relu', padding='same'))(inputs3)
        conv32 = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=11, activation='relu'))(conv3)
        drop3 = layers.TimeDistributed(layers.Dropout(0.5))(conv32)
        pool3 = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(drop3)
        flat3 = layers.TimeDistributed(layers.Flatten())(pool3)

        merged = layers.Concatenate()([flat1, flat2, flat3])
        bilstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(merged)
        bilstm2 = layers.Bidirectional(layers.LSTM(32))(bilstm1)

        dense1 = layers.Dense(128, activation='relu')(bilstm2)
        batchnorm = layers.BatchNormalization()(dense1)
        outputs = layers.Dense(6, activation='softmax')(batchnorm)

        model = models.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
        # plot_model(model, show_shapes=True, to_file="multichannel_uci.png")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        history = model.fit([trainX, trainX, trainX], trainy, epochs=epochs, validation_split=0.3, verbose=1, shuffle=True, batch_size=batch_size)

        model.save(model_name, save_format='h5')

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

    _, accuracy = model.evaluate([testX, testX, testX], testy, batch_size=batch_size, verbose=0)
    # _, accuracy = model.evaluate([trainX, trainX, trainX], trainy, batch_size=batch_size, verbose=0)

    print("Accuracy is: ", accuracy)
    reduced_data, orig_labels, new_labels = cluster(model, mode=2, data=[trainX, trainX, trainX], labels=trainy, layer=4)
    visual = Visual(np.array(reduced_data), np.array(orig_labels))
    # visual.densmap(neighbours=300, dist=0.01)
    visual.tmap(100, 10)
    # visual.clusterplot()

elif x == 2:
    print("Starting ConvAE Model")
    trainX, trainy, testX, testy = load_dataset()
    trainX, testX = scale_data(trainX, testX, True)
    n_timestamps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    epochs, batch_size = 100, 128
    n_steps, n_length = 4, 32
    SAVE_LOCATION = "models/autoencoder/convae_uci_test"

    def create_model():
        input = layers.Input(shape=(n_timestamps, n_features))
        x = layers.Reshape((n_steps, n_length, n_features))(input)

        # Encoder
        x = layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu"))(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, strides=2))(x)

        # Decoder
        x = layers.TimeDistributed(layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, activation="relu"))(x)

        x = layers.TimeDistributed(layers.Flatten())(x)
        # LSTM
        x = layers.Bidirectional(layers.LSTM(100, recurrent_dropout=0.5))(x)
        # x = layers.LSTM(150, recurrent_dropout=0.4)(x)

        # Dense
        x = layers.Dense(300, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(100, activation="relu")(x)
        x = layers.Dense(6, activation="softmax")(x)

        model = models.Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        model.summary()
        return model

    model = create_model()
    assert False
    try:
        print("Looking for UCI model")
        model.load_weights(SAVE_LOCATION + '/')
        # uci_model.summary()
    except:
        model.fit(trainX, trainy, epochs=epochs, validation_split=0.3, batch_size=batch_size, verbose=1)
        if not os.path.exists(SAVE_LOCATION):
            os.makedirs(SAVE_LOCATION)

        model.save_weights(SAVE_LOCATION + '/')

    _, test_acc = model.evaluate(testX, testy, verbose=0, batch_size=batch_size)
    print("Test accuracy is ", test_acc)

    cluster(model, trainX, trainy, 2)

elif x == 3:

    def root_mean_log_squared_error(y_true, y_pred):
        # n = len(y_true)
        # msle = np.mean([(np.log(y_pred[i] + 1) - np.log(y_true[i] + 1)) ** 2.0 for i in range(n)])
        # return np.sqrt(msle)
        # return np.sqrt(mean_squared_log_error(y_true, y_pred))
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.nn.relu(y_pred)
        return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(tf.math.log1p(y_pred), tf.math.log1p(y_true))))

    def repeat(x_inp):
        x, inp = x_inp
        x = tf.expand_dims(x, 1)
        x = tf.repeat(x, [tf.shape(inp)[1]], axis=1)
        return x

    def create_model(timestamps, features):
        input_layer = layers.Input(shape=(None, features))
        # masking = layers.Masking(mask_value=0)(input_layer)
        encode = layers.Conv1D(128, kernel_size=5, activation="relu")(input_layer)
        encode = layers.LSTM(64, return_sequences=False)(encode)
        # encode = layers.LSTM(16, return_sequences=False)(encode)

        # code = layers.Lambda(repeat)([encode, input_layer])
        code = layers.RepeatVector(timestamps)(encode)

        # decode = layers.LSTM(16, return_sequences=True)(code)
        decode = layers.LSTM(64, return_sequences=True)(code)
        output_layer = layers.TimeDistributed(layers.Dense(features, activation="sigmoid", dtype='float32'))(decode)
        
        lstm_auto = models.Model(inputs=input_layer, outputs=output_layer)
        lstm_auto.compile(optimizer='adam', loss=root_mean_log_squared_error)
        return lstm_auto

    print("Starting LSTM autoencoder")
    model_name = "models/autoencoder/lstm_auto_uci_16"

    trainX, trainy, testX, testy = load_dataset()
    trainX = np.concatenate((trainX, testX))
    trainy = np.concatenate((trainy, testy))
    trainX, _ = scale_data(trainX, trainX, True)
    n_timestamps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = create_model(n_timestamps, n_features)
    try:
        model.load_weights(model_name)
    except:
        print("Begin model creation")
        history = model.fit(trainX, trainX, epochs=300, verbose=1, batch_size=128, validation_split=0.2)
        model.save_weights(model_name)
    else:
        print("Model loaded in!")

    encode = models.Model(inputs=model.inputs, outputs=model.layers[-4].output)
    encoded_data = encode.predict(trainX, batch_size=128)
    true_labels = [y.argmax() for y in trainy]

    # encoded_test = encode.predict(testX, batch_size=128)
    # test_labels = [y.argmax() for y in testy]
    # train_data = np.concatenate((encoded_data, encoded_test))
    # train_labels = np.concatenate((true_labels, test_labels))


    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette")
    kmeans = KMeans(init="k-means++", n_clusters=n_outputs, n_init=4)
    bench_k_means(kmeans=kmeans, name="k-means++", data=encoded_data, labels=true_labels)
    print(82 * "_")

    x_train, x_test, y_train, y_test = model_selection.train_test_split(encoded_data, trainy, test_size=0.2)

    classify_features = x_train.shape[1]
    print("MLP Classifier")
    classify = models.Sequential()
    classify.add(layers.Dense(500, activation='relu', input_shape=(classify_features,)))
    # classify.add(layers.Dropout(0.5))
    classify.add(layers.BatchNormalization())
    classify.add(layers.Dense(100, activation='relu'))
    classify.add(layers.Dense(6, activation='softmax'))
    classify.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = classify.fit(x_train, y_train, epochs=150, verbose=1, shuffle=True, validation_split=0.2)

    _, test_acc = classify.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy is ", test_acc)

    print("\n")
    print("Testing LSTM classifier")
    x_train = x_train.reshape(-1, 1, classify_features)
    x_test = x_test.reshape(-1, 1, classify_features)
    # y_train = trainy
    # y_test = testy

    input_layer = layers.Input(shape=(None, classify_features))
    hidden_layer = layers.Bidirectional(layers.LSTM(100, return_sequences=True))(input_layer)
    hidden_layer = layers.Bidirectional(layers.LSTM(50))(hidden_layer)
    hidden_layer = layers.Dense(100, activation='relu')(hidden_layer)
    hidden_layer = layers.Dropout(0.5)(hidden_layer)
    output_layer = layers.Dense(6, activation='softmax')(hidden_layer)
    lstm_classifier = models.Model(inputs=input_layer, outputs=output_layer)
    lstm_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = lstm_classifier.fit(x_train, y_train, epochs=150, verbose=1, shuffle=True, validation_split=0.2)

    _, test_acc = lstm_classifier.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy is ", test_acc)

elif x == 4:
    print("Base TE")
    def create_model():
        model = models.Sequential()
        model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=(None, 9)))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(6, activation='softmax'))
        model.summary()
        # optimizing = optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
        optimizing = optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizing, metrics=['accuracy'])
        return model
    trainX, trainy, testX, testy = load_dataset()
    trainX, testX = scale_data(trainX, testX, True)
    # trainX = trainX.reshape(-1, trainX.shape[2])
    # testX =  testX.reshape(-1, testX.shape[2])

    model = create_model()
    history = model.fit(trainX, trainy, epochs=64, verbose=1, batch_size=64, validation_split=0.2, shuffle=False)
    _, test_acc = model.evaluate(testX, testy, verbose=0, batch_size=64)
    print("Accuracy of final model is:", test_acc)

elif x == 5:
    print("TimeDist UCI")
    trainX, trainy, testX, testy = load_dataset()
    trainX, testX = scale_data(trainX, testX, True)
    n_features, n_timesteps = trainX.shape[2], trainX.shape[1]
    
    segments = 4
    trainX = trainX.reshape(-1, segments, n_timesteps // segments, n_features)
    testX = testX.reshape(-1, segments, n_timesteps // segments, n_features)

    batch_size, epochs = 64, 64
    def create_model():
        model = models.Sequential()
        # model.add(layers.TimeDistributed(layers.Masking(), input_shape=(None, n_steps, n_time, n_features)))
        model.add(layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=5, activation='relu'), input_shape=(None, n_timesteps // segments, n_features)))
        model.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dropout(0.5)))
        model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
        model.add(layers.TimeDistributed(layers.Flatten()))
        # model.add(layers.TimeDistributed(layers.Masking(mask_value=0)))
        model.add(layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.5)))
        # model.add(layers.Bidirectional(layers.LSTM(32, recurrent_dropout=0.5)))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(6, activation='softmax'))
        # model.summary()

        optimizing = optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizing, metrics=['accuracy'])
        return model
    model_name = "models/timedis/uci"
    try:
        print("looking for UCI time dist weights")
        model = models.load_model(model_name)
    except:
        print("model not found, creating model")
        model = create_model()
        history = model.fit(trainX, trainy, epochs=epochs, verbose=1, batch_size=batch_size, validation_split=0.2, shuffle=True)
        model.save(model_name, save_format='h5')

    _, test_acc = model.evaluate(testX, testy, verbose=0, batch_size=batch_size)
    print("Accuracy of final model is:", test_acc)

    reduced_data, orig_labels, new_labels = cluster(model, mode=2, data=trainX, labels=trainy, layer=4)
    visual = Visual(np.array(reduced_data), np.array(orig_labels))
    # visual.densmap(neighbours=300, dist=0.01)
    visual.tmap(100, 10)
    # visual.clusterplot()

elif x == 6:
    print("Improved TE")
    trainX, trainy, testX, testy = load_dataset()
    assert False
    trainX, testX = scale_data(trainX, testX, True)
    n_features, n_timesteps = trainX.shape[2], trainX.shape[1]

    # segments = 4
    # trainX = trainX.reshape(-1, segments, n_timesteps // segments, n_features)
    # testX = testX.reshape(-1, segments, n_timesteps // segments, n_features)

    batch_size, epochs = 64, 64
    def create_model():
        cnn_lstm = models.Sequential()
        cnn_lstm.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(None, n_features)))
        cnn_lstm.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        cnn_lstm.add(layers.LSTM(128, return_sequences=True))
        cnn_lstm.add(layers.LSTM(64, return_sequences=False))
        cnn_lstm.add(layers.Dense(128, activation='relu'))
        cnn_lstm.add(layers.Dropout(0.5))
        cnn_lstm.add(layers.Dense(6, activation='softmax'))
        cnn_lstm.summary()
        optimizing = optimizers.Adam(learning_rate=0.001)
        cnn_lstm.compile(loss='categorical_crossentropy', optimizer=optimizing, metrics=['accuracy'])
        return cnn_lstm
    
    model = create_model()
    history = model.fit(trainX, trainy, epochs=epochs, verbose=1, batch_size=batch_size, validation_split=0.2, shuffle=True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    _, test_acc = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    print("Accuracy of final model is:", test_acc)

