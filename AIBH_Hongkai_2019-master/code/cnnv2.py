# CNN version2
import os 

import time
import numpy as np
import tensorflow as tf
import datapkg
import pandas as pd

from tensorflow.keras import layers, models


#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession


MODEL_SEVE_PATH = 'model/cnn'
TRAIN_PATH = "merged/train.csv"
TEST_PATH = "merged/test.csv"

Label = {1:'STD',2:'WAL',3:'JOG',4:'JUM',5:'STU',6:'STN',7:'SCH',8:'SIT',9:'CHU',10:'CSI',11:'CSO',12:'LYI'}
#Label = {1:'STD',2:'WAL',3:'JOG',4:'JUM',5:'STU',6:'STN',7:'SCH',8:'SIT',9:'CHU',10:'CSI',11:'CSO',12:'LYI',20:'FOL',21:'FKL',22:'BSC',23:'SDL'}
# Hyper parameter
CLASS_LIST = [1,2,3,4,5,6,7,8,9,10,11]
#CLASS_LIST = [1,2,3,4,5,6,7,8,9,10,11,12,20,21,22,23]

CLASS_NUM = len(CLASS_LIST)
LEARNING_RATE = 0.001
TRAIN_STEP = 80000
BATCH_SIZE = 32

def wights_tensor(shape):
    wights = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(wights,dtype=tf.float32)


def biases_tensor(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias,dtype=tf.float32)

#convolution layer
def conv2d(x,kernel):
    return tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding='SAME')

# maxpooling layer
def max_pooling_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# lrn layer
def lrn(x):
    return tf.nn.lrn(x,4,1.0,0.001,0.75)

def HARnet(x):

    with tf.name_scope('reshape'):
        x = tf.reshape(x,[-1,18,18,3])
        x = x / 255.0 * 2 - 1

    with tf.name_scope('conv1'):
        # output:[-1,18,18,16]
        conv1_kernel = wights_tensor([5,5,3,32])
        conv1_bias = biases_tensor([32])
        conv1_conv = conv2d(x,conv1_kernel)+conv1_bias
        conv1_value = tf.nn.relu(conv1_conv)

    with tf.name_scope('max_pooling_1'):
        # output:[-1,10,10,16]
        mp1 = max_pooling_2x2(conv1_value)

    with tf.name_scope('conv2'):
        # output:[-1,8,8,64]
        conv2_kernel = wights_tensor([5,5,32,64])
        conv2_bias = biases_tensor([64])
        conv2_conv = conv2d(mp1,conv2_kernel)+conv2_bias
        conv2_value = tf.nn.relu(conv2_conv)

    with tf.name_scope('max_pooling_2'):
        # output:[-1,5,5,64]
        mp2 = max_pooling_2x2(conv2_value)

    with tf.name_scope('fc1'):
        fc1_wights = wights_tensor([5*5*64,512])
        fc1_biases = biases_tensor([512])

        fc1_input = tf.reshape(mp2,[-1,5*5*64])
        fc1_output = tf.nn.relu(tf.matmul(fc1_input,fc1_wights)+fc1_biases)

    with tf.name_scope('drop_out'):
        keep_rate = tf.placeholder(dtype=tf.float32)
        drop_out = tf.nn.dropout(fc1_output,keep_rate)

    with tf.name_scope('fc2'):
        fc2_wights = wights_tensor([512,CLASS_NUM])
        fc2_biases = biases_tensor([CLASS_NUM])
        fc2_output = tf.matmul(drop_out,fc2_wights)+fc2_biases

    return fc2_output,keep_rate


# def train_model():
#     print ("start train at :", time.asctime( time.localtime(time.time()) ))
#     with tf.name_scope('input_dataset'):
#         x = tf.placeholder(tf.float32,[None,972])
#         y = tf.placeholder(tf.float32,[None,CLASS_NUM])
#     print ("check A :", time.asctime( time.localtime(time.time()) ))
#     y_,keep_rate = HARnet(x)
#     print ("check B :", time.asctime( time.localtime(time.time()) ))
#     with tf.name_scope('loss'):
#         cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)
#         loss = tf.reduce_mean(cross_entropy)
#         tf.summary.scalar("loss", loss)
#     print ("check C :", time.asctime( time.localtime(time.time()) ))
#     with tf.name_scope('optimizer'):
#         train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
#     print ("check D :", time.asctime( time.localtime(time.time()) ))
#     with tf.name_scope('accuracy'):
#         correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
#         correct_prediction = tf.cast(correct_prediction,tf.float32)
#         accuracy = tf.reduce_mean(correct_prediction)
#         tf.summary.scalar("accuracy", accuracy)
#     print ("check E :", time.asctime( time.localtime(time.time()) ))
#     data = datapkg.DataPKG(TRAIN_PATH, TEST_PATH, CLASS_LIST)
#     saver = tf.train.Saver()
#     merged = tf.summary.merge_all()
#     start_time = time.time()
#     print ("check F :", time.asctime( time.localtime(time.time()) ))
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         train_writer = tf.summary.FileWriter("log/", sess.graph)
#         print ("check G :", time.asctime( time.localtime(time.time()) ))
#         for step in range(1, TRAIN_STEP+1):
#             print ("start loading new batch :", time.asctime( time.localtime(time.time()) ))
#             batch_x, batch_y = data.next_batch(BATCH_SIZE)
#             print ("finish loading new batch :", time.asctime( time.localtime(time.time()) ))
#             print(step)
#             if step%100==0:
#                 train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_rate: 1.0})
#                 print('NO. %d training, accuracy is: %f' % (step, train_accuracy))
#                 summ = sess.run(merged, feed_dict={x: batch_x, y: batch_y,keep_rate: 1.0})
#                 train_writer.add_summary(summ, global_step=step)
            
#             print ("start feeding new batch :", time.asctime( time.localtime(time.time()) ))
#             train.run(feed_dict={x: batch_x, y: batch_y, keep_rate: 0.5})
#             print ("finish feeding new batch :", time.asctime( time.localtime(time.time()) ))

#         train_writer.close()
#         save_path = saver.save(sess, MODEL_SEVE_PATH)
#         print("task done, this model has been saved to the following path: %s"%(save_path))
        
#     train_time = str(time.time() - start_time)
#     print('train_time is',train_time)

# def test_model():
#     data = datapkg.DataPKG(TRAIN_PATH, TEST_PATH, CLASS_LIST)
#     test_x, test_y = data.get_test_data()

#     tf.reset_default_graph()
#     with tf.name_scope('input'):
#         x = tf.placeholder(tf.float32,[None,972])
#         y = tf.placeholder(tf.float32,[None,CLASS_NUM])
#     y_,keep_rate = HARnet(x)

#     with tf.name_scope('accuracy'):
#         correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
#         correct_prediction = tf.cast(correct_prediction,tf.float32)
#         accuracy = tf.reduce_mean(correct_prediction)

#     start_time = time.time()

#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, MODEL_SEVE_PATH)
#         p_y = np.argmax(sess.run(y_,feed_dict={x: test_x,keep_rate: 1.0}),1)
#         print("Oveall_Accuracy is: %f" % accuracy.eval(feed_dict={x: test_x, y: test_y, keep_rate: 1.0}))

#     test_time = str(time.time() - start_time)
#     print('test_time is ï¼š',test_time)

#     g_truth = np.argmax(test_y,1)


#     for i in range(CLASS_NUM):
#         accuracy,sensitivity,specificity = evaluate(p_y,g_truth,i)
#         print('For label:%5s, accuracy is: %04f (sensitivity:%04f, specificity:%04f)'%(Label[CLASS_LIST[i]],accuracy,sensitivity,specificity))
        
def create_model():
    model = models.Sequential()
    model.add(layers.Reshape((18, 18, 3), input_shape=(972,)))

    model.add(layers.Lambda(lambda x: x / 255))
    model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
    model.add(layers.MaxPool2D((2, 2), strides=2))
    # model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
    model.add(layers.MaxPool2D((2,2), strides=2))
    # model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
    model.add(layers.MaxPool2D((2,2), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(11, activation="softmax")) # The number of final classes
    return model

def evaluate(p,g,class_):
    match_idx = []
    data_size = g.size
    for i in range(data_size):
        if g[i] ==class_:
            match_idx.append(i)
    match_num = len(match_idx)

    TP = 0
    FN = 0
    for i in range(match_num):
        index = match_idx[i]
        if p[index] == g[index]:
            TP+=1
        else:
            FN+=1
    sensitivity = TP/(TP+FN)


    FP =0
    TN =0
    for i in range(data_size):
        if g[i]!=class_:
            if p[i] == class_:
                FP+=1
            else:
                TN+=1

    specificity = TN/(FP+TN)

    accuracy = (TP+TN)/(FP+TN+TP+FN)
    return accuracy,sensitivity,specificity


if __name__=='__main__':
    #config = ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)
    # train_model()
    # test_model()
    data = datapkg.DataPKG(TRAIN_PATH, TEST_PATH, CLASS_LIST)
    trainx, trainy = data.get_train_data()
    testx, testy = data.get_test_data()

    # trainx = trainx / 255 * 2.0 - 1

    epochs = 64
    cnn = create_model()
    cnn.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
    cnn.summary()
    cnn.fit(trainx, trainy, epochs=epochs, verbose=1, batch_size=64, validation_split=0.2, shuffle=True)
    
    _, test_acc = cnn.evaluate(testx, testy, batch_size=64)
    print("Testing accuracy is ", test_acc)
    

