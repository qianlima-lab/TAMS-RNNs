import tensorflow as tf
import numpy as np
import random
import math
from functools import reduce
from operator import mul
import sys
import cPickle
sys.path.append("..")
from TAMSRNN import *
import os
import reader as reader
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def one_flod(train_data, train_labels, test_data, test_labels):
    graph = tf.Graph()
    with graph.as_default():
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        m = model(train_data, num_class, parameter_configs)
        with tf.Session(config = tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            print tf.trainable_variables ()
            data_generate_train = reader.next_batch_data(train_data, train_labels, batch_size)
            data_generate_test = reader.next_batch_data(test_data, test_labels, test_batch_size)
            max_test_accuracy = 0
            for step in range(epoch * 200):
                input_x, input_y = data_generate_train.next()
                feed_dict = {m.x: input_x,
                             m.y: input_y,
                             m.droup_keep_prob: 1.0,
                             m.x_keep_prob:1.0,
                             m.on_train: True
                             }
                train_step,  acc, loss = sess.run([m.train_step, m.acc, m.cross_entropy], feed_dict)

                if step % 200 == 0:
                    print "train_step: %d, train_acc: %f, loss: %f"%(step,acc,loss)
                    test_acc_nums = []
                    true_label_list = []
                    predict_label_list = []
                    dw_list = np.zeros((len(test_data), test_data.shape[1], 4 * len(period)))
                    
                    for test_batch_i in range(test_batch_nums):
                        test_x, test_y = data_generate_test.next()
                        test_accuracy_i, test_true_label, test_predict_label, dynamic_weight = sess.run([m.acc, m.true_label, m.prediction_label, m.dynamic_weight], 
                                                                               feed_dict={m.x: test_x, m.y: test_y, m.droup_keep_prob: 1.0, m.x_keep_prob:1.0, m.on_train: False})
                        test_acc_nums.append(test_accuracy_i * test_x.shape[0])
                        for i in range(len(test_true_label)):
                            true_label_list.append(test_true_label[i])
                            predict_label_list.append(test_predict_label[i])
                        dw_list[test_batch_i * test_batch_size: (test_batch_i+1) * test_batch_size] = dynamic_weight
                            
                    test_accuracy = sum(test_acc_nums) / (test_nums + 0.0)
                    print "test_acc:", test_accuracy,
                    if test_accuracy > max_test_accuracy:
                        max_test_accuracy = test_accuracy
                    print "max_test_acc:", max_test_accuracy
                    
    return max_test_accuracy
    
    
if __name__ == '__main__':
    
    time_length = 6000
    num_class = 8
    batch_size = 32
    test_batch_size = 80
    period = [1, 4, 16, 64]
    epoch = 200
    parameter_configs = {"learning_rate":0.001,
                         "lstm_hidden_size":192,
                         "temperature":1,
                         "period": [1, 4, 16, 64]
                          }   
                          
    print('Loading data...')
    #----------------------------data------------------------------------------
    train_filepath = './dataset/train_data.p'
    val_filepath = './dataset/val_data.p'
    test_filepath = './dataset/test_data.p'
    train_data, train_labels = reader.load_data(train_filepath , time_length ,  1 )
    val_data, val_labels = reader.load_data(val_filepath, time_length ,  1)
    test_data, test_labels = reader.load_data(test_filepath, time_length ,  1)
    
    print train_data.shape
    print val_data.shape
    print test_data.shape
    
    test_nums = test_data.shape[0]
    test_batch_nums = int(math.ceil(test_nums / (test_batch_size + 0.0)))

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

    print "max_test_acc:", one_flod(train_data, train_labels, test_data, test_labels)
