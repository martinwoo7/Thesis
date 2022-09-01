import os
# os.environ["OMP_NUM_THREADS"] = "4"
import time
import h5py 

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from scipy.spatial import distance as dst
from sklearn import model_selection
from sklearn import preprocessing as preproc
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score

from tensorflow.keras import models, optimizers, preprocessing, layers
from tensorflow.keras import backend as K
from flexible_clustering import FISHDBC
from itertools import islice

from te_utils import *
from uci_utils import *
# from dtw import *

def distance(x, y):
    return np.linalg.norm(x - y)
    # return dst.minkowski(x, y, 1)
    # return dst.cityblock(x, y)
    # return dtw(x, y, distance_only=True).distance

def _input(message, input_type=str):
    while True:
        try:
            a = input_type(input(message))
            if a not in range(0, 4):
                raise ValueError
            return a
        except ValueError:
            print("Provide an acceptable input")

def repeat(x_inp):
        x, inp = x_inp
        x = tf.expand_dims(x, 1)
        x = tf.repeat(x, [tf.shape(inp)[1]], axis=1)
        return x

def grouper(batch_size, iterable):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch
    # return list(zip(*[iter(iterable)] * n))

def cluster(model, data=None, labels=None, mode=1):
    '''
    Takes in a CNN_LSTM model
    '''
    print("Clustering")
    popped_model = models.Model(inputs=model.input, outputs=model.layers[-5].output)
    # popped_model.summary()

    if mode == 1:
        EF = 50
        MIN_SAMPLES = 1
        MIN_CLUSTER_SIZE = 100 # None
        # CLUSTER_SELECTION_EPSILON = 1

        true_labels = []
        # test_labels = []
        fishdbc = FISHDBC(distance, ef=EF, min_samples=MIN_SAMPLES)

        # test_outputs = model.predict([data, data, data])
        test_outputs = model.predict(data)
        test_outputs = [y.argmax() for y in test_outputs]
        test_labels = [y.argmax() for y in labels]

        differences = [test_outputs[i]==test_labels[i] for i in range(len(test_labels))]
        difference_i = [i for i, value in enumerate(differences) if not value]  

        del test_outputs
        data = np.delete(data, difference_i, axis=0)
        print(data.shape)

        test_labels  = [test_labels[i] for i, value in enumerate(differences) if value]

        # cnn_output = popped_model.predict([data, data, data])
        cnn_output = popped_model.predict(data)
        true_labels = test_labels

        del differences
        del difference_i
        del test_labels
        test_output = preproc.Normalizer().fit_transform(cnn_output)

        print(82 * "_")
        print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette") 
        t0 = time.time()

        # Want to update in increments probably
        for batch in grouper(400, test_output):
            fishdbc.add(np.array(batch))
        # labs, _, _, ctree, _, _ = fishdbc.cluster(min_cluster_size=MIN_CLUSTER_SIZE, cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON)
        labs, _, _, _, _, _ = fishdbc.cluster(min_cluster_size=MIN_CLUSTER_SIZE)

        fit_time = time.time() - t0
        results = ["FISHDBC", fit_time, 0]

        true_labels = np.array(true_labels)

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
        print(len(values))
        if len(values) > 14 and len(values) < 18:
            print(values)
            print(counts)
        print(82 * "_")
        
    
    elif mode == 2:
        # test_outputs = model.predict([data, data, data])
        test_outputs = model.predict(data)
        test_outputs = [y.argmax() for y in test_outputs]
        test_labels = [y.argmax() for y in labels]

        differences = [test_outputs[i]==test_labels[i] for i in range(len(test_labels))]
        difference_i = [i for i, value in enumerate(differences) if not value]  

        data = np.delete(data, difference_i, axis=0)
        print(data.shape)

        true_labels  = [test_labels[i] for i, value in enumerate(differences) if value]

        # cnn_output = popped_model.predict([data, data, data])
        cnn_output = popped_model.predict(data)
        test_output = preproc.Normalizer().fit_transform(cnn_output)

        print(82 * "_")
        print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette") 
        kmeans = KMeans(init="k-means++", n_clusters=16, n_init=4)
        bench_k_means(kmeans=kmeans, name="k-means++", data=test_output, labels=true_labels)
        print(82 * "_")

    else:
        print("Mode not found. Clustering skipped")

x = _input("(0) Skip, (1) LSTM AE, (2) ConvAE, (3) Test model", int)            

if x == 0:
    print("skipped")

if x == 1:
    # LSTM Autoencoder
    # segmented3 - many to one w/ overlap and no addition augmentations - deleted
    # segmented_test - many to many w/o feature engineering
    def distance(x, y):
        return dst.euclidean(x, y)
        # return dst.minkowski(x, y, 1)
        # return dst.cityblock(x, y)
        # return dtw(x, y, distance_only=True).distance

    def reduce_gen(state=42, validate=False):
        while True:
            for i in range(15):
                action = getDataset("reduced/" + str(i) + ".hdf5", str(i))
                label = getDataset("reduced/" + str(i) + "_labels.hdf5", str(i) +"_labels")
                # print(action.shape, label.shape)
                x_train, x_test, y_train, y_test = model_selection.train_test_split(action, label, test_size=0.2, random_state=state)
                if validate:
                    yield x_test, y_test
                else:
                    yield x_train, y_train

    def create_model():
        input_layer = layers.Input(shape=(None, features))
        # masking = layers.Masking(mask_value=0)(input_layer)
        encode = layers.Conv1D(128, kernel_size=5, activation="relu")(input_layer)
        encode = layers.LSTM(64, return_sequences=False)(encode)

        # code = layers.Lambda(repeat)([encode, input_layer])
        code = layers.RepeatVector(200)(encode)

        decode = layers.LSTM(64, return_sequences=True)(code)
        output_layer = layers.TimeDistributed(layers.Dense(features, activation="sigmoid"))(decode)
        
        lstm_auto = models.Model(inputs=input_layer, outputs=output_layer)
        lstm_auto.compile(optimizer='adam', loss='mse')
        return lstm_auto

    print("Starting LSTM autoencoder")
    model_name = "models/autoencoder/lstm_auto_seq"
    # dataset_loc = "datasets/stacked/"
    # trainx = getDataset(dataset_loc + "stacked_training_data.hdf5", "stacked_training_dataset")
    # testx = getDataset(dataset_loc + "stacked_testing_data.hdf5", "stacked_testing_dataset")
    DATASET = "raw"
    trainX, trainy = load_train(DATASET)
    testX, testy = load_test(DATASET)
    
    scaler = preproc.Normalizer()
    orig_shape = trainX.shape
    scaler = scaler.fit(trainX.reshape(-1, orig_shape[-1]))
    trainX = scaler.transform(trainX.reshape(-1, orig_shape[-1])).reshape(orig_shape)
    features = orig_shape[-1]

    lstm_auto = create_model()
    try:
        lstm_auto.load_weights(model_name)
    except:
        print("Begin model creation")
        # history = lstm_auto.fit(train_generator2(scaler=scaler, encode=True, group="train"), steps_per_epoch=15, epochs=300, verbose=1, shuffle=True)
        lstm_auto.fit(trainX, trainX, epochs=300, batch_size=256, verbose=1)
        # save the weights instead
        lstm_auto.save_weights(model_name)
    else:
        print("model loaded in!")
    
    # use reduced data folder
    encode = models.Model(inputs=lstm_auto.inputs, outputs=lstm_auto.layers[-4].output)
    dataset = np.concatenate((trainX, testX))
    labels = np.concatenate((trainy, testy))
    output_data = encode.predict(dataset)

    train_data, test_data, train_labels,test_labels = model_selection.train_test_split(output_data, labels, test_size=0.2)

    # gen = train_generator2(scaler=scaler, encode=False, group="train")
    # data_list, label_list = None, None
    # for i in range(15):
    #     data, labels = next(gen)
    #     reduce = encode.predict(data)
    #     if data_list is None:
    #         data_list = reduce
    #         label_list = labels
    #     else:
    #         data_list = np.vstack((data_list, reduce))
    #         label_list = np.vstack((label_list, labels))

    clus_labs = np.array([y.argmax() for y in train_labels])
    n_clusters = len(np.unique(clus_labs))
    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tNMI\tsilhouette")

    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    bench_k_means(kmeans=kmeans, name="k-means++", data=train_data, labels=clus_labs)
    print(82 * "_")


    # FISHDBC
    MIN_SAMPLES = 1
    MIN_CLUSTER_SIZE = 100
    CLUSTER_SELECTION_EPSILON = 0.0

    fishdbc = FISHDBC(distance, min_samples=MIN_SAMPLES)
    fishdbc.update(train_data)
    labs, _, _, _, _, _ = fishdbc.cluster(min_cluster_size=MIN_CLUSTER_SIZE, cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON)
    ari = adjusted_rand_score(labs, clus_labs)
    nmi = normalized_mutual_info_score(labs, clus_labs)
    print("ARI is: ", ari)
    print("NMI is: ", nmi)

    
    classify = models.Sequential()
    classify.add(layers.Dense(1000, activation='relu', input_shape=(128,)))
    # classify.add(layers.Dropout(0.5))
    classify.add(layers.BatchNormalization())
    classify.add(layers.Dense(500, activation='relu'))
    classify.add(layers.Dense(16, activation='softmax'))
    classify.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = classify.fit(train_data, train_labels, epochs=64, verbose=1, shuffle=True, validation_split=0.2, batch_size=256)

    _, test_acc = classify.evaluate(test_data, test_labels, verbose=0, batch_size=64)
    print("Test accuracy is ", test_acc)
    
    print("\n")
    print("Testing LSTM classifier")
    data_list = output_data.reshape(-1, 1, 128)
    label_list = output_data.reshape(-1, 1, 16)
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(data_list, label_list, test_size=0.2)

    input_layer = layers.Input(shape=(None, 128))
    hidden_layer = layers.Bidirectional(layers.LSTM(10, return_sequences=True))(input_layer)
    # hidden_layer = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(hidden_layer)
    hidden_layer = layers.TimeDistributed(layers.Dense(50, activation='relu'))(hidden_layer)
    hidden_layer = layers.Dropout(0.5)(hidden_layer)
    output_layer = layers.TimeDistributed(layers.Dense(16, activation='softmax'))(hidden_layer)
    lstm_classifier = models.Model(inputs=input_layer, outputs=output_layer)
    lstm_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = lstm_classifier.fit(train_data, train_labels, epochs=100, verbose=1, shuffle=True, validation_split=0.2, batch_size=256)

    _, test_acc = lstm_classifier.evaluate(test_data, test_labels, verbose=0, batch_size=128)
    print("Test accuracy is ", test_acc)
    # TODO: Is the performance shit because of feature engineering? Try feeding in raw data w/o engineering to autoencoder

elif x == 2:
    print("Starting ConvAE")

    def create_model():
        input = layers.Input(shape=(200, 9))
        x = layers.Reshape((4, 50, 9))(input)

        # Encoder
        x = layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=5, strides=2, activation="relu"))(x)
        x = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2, strides=2))(x)

        # Decoder
        x = layers.TimeDistributed(layers.Conv1DTranspose(filters=128, kernel_size=5, strides=2, activation="relu"))(x)

        x = layers.TimeDistributed(layers.Flatten())(x)
        # LSTM
        # x = layers.LSTM(100, recurrent_dropout=0.4)(x)
        x = layers.Bidirectional(layers.LSTM(50, recurrent_dropout=0.5, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(20, recurrent_dropout=0.5))(x)

        # Dense
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(16, activation="softmax")(x)

        model = models.Model(inputs=input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        # model.summary()
        return model
    
    DATASET = "raw"
    
    training_data, training_labels = load_train(DATASET)
    print("original shape is: ", training_data.shape)
    # testing_data, testing_labels = load_test(DATASET)
    # data = np.concatenate((training_data, testing_labels))
    # labels = np.concatenate((testing_data, testing_labels))

    changed_labels = np.array([y.argmax() for y in training_labels])
    data_mask = ma.getmask(ma.masked_where(changed_labels < 12, changed_labels))
    
    training_data = training_data[data_mask]
    # print(np.unique(changed_labels))
    changed_labels = changed_labels[data_mask]
    training_labels = utils.to_categorical(changed_labels, num_classes=12)

    print("final shape is: ", training_data.shape)
    assert False

    # temp = []
    # for obs in training_data:
    #     temp.append(fuse(2, obs))

    # print(np.array(temp).shape)
    # assert False

    orig_shape = training_data.shape
    scaler = preproc.StandardScaler().fit(training_data.reshape(-1, orig_shape[2]))
    training_data = scaler.transform(training_data.reshape(-1, orig_shape[2])).reshape(orig_shape)
    SAVE_LOCATION ="models/autoencoder/convae_raw"
    model = create_model()
    try:
        model.load_weights(SAVE_LOCATION + '/')
    except:
        
        model.fit(training_data, training_labels, epochs=100, validation_split=0.3, batch_size=128, verbose=1)

        if not os.path.exists(SAVE_LOCATION):
            os.makedirs(SAVE_LOCATION)

        model.save_weights(SAVE_LOCATION + '/')

    testing_data, testing_labels = load_test(DATASET)
    # changed_labels = np.array([y.argmax() for y in testing_labels])
    # data_mask = ma.getmask(ma.masked_where(changed_labels < 12, changed_labels))
    
    # testing_data = testing_data[data_mask]
    # changed_labels = changed_labels[data_mask]
    # testing_labels = utils.to_categorical(changed_labels, num_classes=12)
    # # testing_data = fuse(2, testing_data)

    orig_test = testing_data.shape
    testing_data = scaler.transform(testing_data.reshape(-1, orig_test[2])).reshape(orig_test)
    _, test_acc = model.evaluate(testing_data, testing_labels, verbose=0)
    print("Test accuracy is ", test_acc)
    cluster(model, training_data, training_labels, mode=2)

elif x == 3:

    def temporalize(X, y, lookback):
        output_X = []
        output_y = []
        for i in range(len(X)-lookback-1):
            t = []
            for j in range(1,lookback+1):
                # Gather past records upto the lookback period
                t.append(X[[(i+j+1)], :])
            output_X.append(t)
            output_y.append(y[i+lookback+1])
        return output_X, output_y
    timeseries = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       [0.1**3, 0.2**3, 0.3**3, 0.4**3, 0.5**3, 0.6**3, 0.7**3, 0.8**3, 0.9**3]]).transpose()
    timesteps = 3
    X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)

    n_features = 2
    X = np.array(X)
    X = X.reshape(X.shape[0], timesteps, n_features)

    model = models.Sequential()
    # model.add(layers.LSTM(128, input_shape=(timesteps,n_features), return_sequences=True))
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps, n_features)))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=False)))

    model.add(layers.RepeatVector(timesteps))

    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.TimeDistributed(layers.Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    # model.summary()
    # fit model
    model.fit(X, X, epochs=500, batch_size=5, verbose=0)
    # demonstrate reconstruction
    yhat = model.predict(X, verbose=0)
    print('---Predicted---')
    print(np.round(yhat,3))
    print('---Actual---')
    print(np.round(X, 3))