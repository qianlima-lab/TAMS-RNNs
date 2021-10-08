import tensorflow as tf
import numpy as np
import random
import math
from functools import reduce
from operator import mul
import cPickle as cp
import sys
sys.path.append("..")
from TAMSRNN import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def one_flod(train_data, train_labels, test_data, test_labels):
    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        m = model(train_data, num_class, parameter_configs)
        with tf.Session(config = tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            print tf.trainable_variables ()
            data_generate_train = load_data(train_data, train_labels, batch_size)
            data_generate_test = load_data(test_data, test_labels, test_batch_size)
            max_test_accuracy = 0
            test_accuracy_list = []
            for step in range(300 * train_batch_nums):
                input_x, input_y = data_generate_train.next()
                feed_dict = {
                             m.x: input_x,
                             m.y: input_y,
                             m.droup_keep_prob: 0.75,
                             m.x_keep_prob:0.9,
                             m.on_train: True
                             }
                train_step,  acc, loss = sess.run([m.train_step, m.acc, m.cross_entropy], feed_dict)

                if step % train_batch_nums == 0:
                    print "train_step: %d, train_acc: %f, loss: %f" % (step, acc, loss)
                    test_acc_nums = []
                    for test_batch_i in range(test_batch_nums):
                        test_x, test_y = data_generate_test.next()
                        test_accuracy_i = m.acc.eval(feed_dict={m.x: test_x, m.y: test_y, m.droup_keep_prob: 1.0,m.x_keep_prob:1.0, m.on_train: False})
                        test_acc_nums.append(test_accuracy_i * test_x.shape[0])
                    test_accuracy = sum(test_acc_nums) / (test_nums + 0.0)
                    test_accuracy_list.append(test_accuracy)
                    print "test_acc:", test_accuracy,
                    if test_accuracy > max_test_accuracy:
                        max_test_accuracy = test_accuracy
                    print "max_test_acc:", max_test_accuracy
                    
    return max_test_accuracy, max(test_accuracy_list[-10:])
    
    
if __name__ == '__main__':
    
    batch_size = 16
    test_batch_size = 16
    parameter_configs = {"learning_rate":0.001,
                         "lstm_hidden_size":256,
                         "period": [1, 2, 4, 8]
                          }   
                                              
    list_dir = ['BasicMotions', 'ArticularyWordRecognition', 'NATOPS', 'HandMovementDirection']
    
    for dataset_name in list_dir:
        #Loading data  
        dataset_name = "./dataset/" + dataset_name
        train_data = np.load(dataset_name + "/X_train.npy")
        test_data = np.load(dataset_name + "/X_test.npy")
        train_labels = np.load(dataset_name + "/y_train.npy")
        test_labels = np.load(dataset_name + "/y_test.npy")
        train_labels, num_class = transfer_labels(train_labels)
        test_labels, _ = transfer_labels(test_labels)
        train_labels = train_labels.reshape(train_labels.shape[0])
        test_labels = test_labels.reshape(test_labels.shape[0])
        print (dataset_name)
        print (train_data.shape)
        print (test_data.shape)
        print (train_labels.shape)
        print (test_labels.shape)
        print (num_class)
            
        #shuffle
        train_data_tmp = np.zeros(train_data.shape)
        train_labels_tmp = np.zeros(train_labels.shape)
        count = 0
        samples_train_nums = train_data.shape[0]
        li = range(samples_train_nums)
        random.shuffle(li)
        for i in li:
            train_data_tmp[count] = train_data[i]
            train_labels_tmp[count] = train_labels[i]
            count += 1
        train_data = train_data_tmp
        train_labels = train_labels_tmp
            
        train_nums = train_data.shape[0]
        train_batch_nums = int(math.ceil(train_nums / (batch_size + 0.0)))
        test_nums = test_data.shape[0]
        test_batch_nums = int(math.ceil(test_nums / (test_batch_size + 0.0)))

        max_accuracy, last_accuracy = one_flod(train_data, train_labels, test_data, test_labels)
        print "max_test_acc:", max_accuracy
        print "last_test_acc:", last_accuracy
        print (dataset_name)
           
