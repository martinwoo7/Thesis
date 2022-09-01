import os
import h5py

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection

from tensorflow.keras import layers, models, utils, optimizers
from tensorflow.keras import backend as K

def getDataset(filename, dataset_name):
    hf = h5py.File(filename, "r")
    temp = np.array(hf.get(dataset_name))
    hf.close()
    return temp

class ClusteringLayer(layers.Layer):
    '''
    Clustering layer converts input sample to soft label. Probability is calculated with student's t-dsitribution

    # Example
    model.add(ClusteringLayer(n_clusters=10))

    # Arguments
        n_clusters: number of clusters
        weights: list of numpy array with shape (n_clusters, n_features) which represent initial cluster centers
        alpha: degrees of freedom parameter for student t-distribution
    
    # Input shape
        2D tensor with shape: (n_samples, n_features)

    # Output shape
        2D tensor with shape: (n_samples, n_clusters)
    '''
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = layers.InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    
    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

data = getDataset("stacked_training_data.hdf5", "stacked_training_dataset")
labels = getDataset("stacked_training_labels.hdf5", "stacked_training_labels")
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2, random_state=42)

hidden_size = 32
input_layer = layers.Input(shape=(82, ))
encoding = layers.Dense(200, activation='relu')(input_layer)
encoding = layers.Dense(100, activation='relu')(encoding)
reduced= layers.Dense(hidden_size, activation='relu')(encoding)
decoding = layers.Dense(100, activation='relu')(reduced)
decoding = layers.Dense(200, activation='relu')(decoding)
output_layer = layers.Dense(82, activation='sigmoid')(decoding)

encoder = models.Model(inputs=input_layer, outputs=reduced)

autoencoder = models.Model(inputs=input_layer, outputs=output_layer)

autoencoder.compile(optimizer="adam", loss="mse")
history = autoencoder.fit(x_train, x_train, epochs=300, verbose=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('AutoEncoder Encoding Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

clustering_layer = ClusteringLayer(16, name="clustering")(encoder.output)
dec = models.Model(inputs=encoder.input, outputs=clustering_layer)
dec.compile(optimizer=optimizers.SGD(0.01, 0.9), loss='kld')

kmeans = KMeans(n_clusters=16, n_init=3)
y_pred_last = kmeans.fit_predict(encoder.predict(x_train))
print("K-Means Clustering ", metrics.adjusted_rand_score(y_pred_last, y_train))
dec.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

q = dec.predict(x_train, verbose=0)
p = target_distribution(q)
y_pred = q.argmax(1)
acc = np.round(metrics.accuracy_score(y_train, y_pred), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y_train, y_pred), 5)
ari = np.round(metrics.adjusted_rand_score(y_train, y_pred), 5)
print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))



