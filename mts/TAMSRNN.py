# -*- coding: utf-8 -*-

import tensorflow as tf
import math
import numpy as np
import rnn_cell_impl
 
def load_data(data, label, batch_size):
    while 1:
        batch_num = int(math.ceil(data.shape[0] / (batch_size+0.0)))
        for i in range(batch_num):
            if i != batch_num - 1:
                data_batch = data[i * batch_size: (i + 1) * batch_size]
                label_batch = label[i * batch_size: (i + 1) * batch_size]
            else:
                data_batch = data[i * batch_size:]
                label_batch = label[i * batch_size:]
            yield [data_batch, label_batch]

def transfer_labels(labels):
	indexes = np.unique(labels)
	num_classes = indexes.shape[0]
	num_samples = labels.shape[0]
	for i in range(num_samples):
		new_label = np.argwhere(indexes == labels[i])[0][0]
		labels[i] = new_label
	return labels, num_classes

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
class model(object):
    def __init__(self, data, num_class, parameters):
        self.batch_size = None
        self.n_steps = data.shape[1]
        self.input_dim = data.shape[2]
        self.num_class = num_class

        self.learning_rate = parameters["learning_rate"] 
        self.lstm_hidden_size = parameters["lstm_hidden_size"]
        self.period = parameters["period"]
        self.on_train = tf.placeholder(tf.bool,[],"whether_train")
        global_step = tf.Variable(0, trainable=False)
        
        with tf.name_scope("input"):
            self.droup_keep_prob = tf.placeholder(tf.float32,[],"droup_keep_prob")
            self.x_keep_prob = tf.placeholder(tf.float32,[],"x_keep_prob")
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.input_dim], "input")
            self.x_drop = tf.nn.dropout(self.x, self.x_keep_prob)
            self.y = tf.placeholder(tf.int32, [self.batch_size], 'labels')
            self.y_one_hot = tf.one_hot(self.y, self.num_class)
        
        self.lstm_outputs = []  
        self.lstm_outputs2 = [] 
                
        with tf.name_scope('lstm1'): 
            with tf.variable_scope('lstm1'):
                self.lstm_input = self.x_drop
                lstm_cell = rnn_cell_impl.TAMSRNN_CELL(self.lstm_hidden_size, self.period, forget_bias=1.0)
                init_state = lstm_cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
                state = init_state
                for time_step in range(self.n_steps):
                    if time_step > 0 : tf.get_variable_scope().reuse_variables() 	
                    (cell_h, state, _) = lstm_cell(self.lstm_input[:, time_step, :], state, time_step)
                    cell_output = cell_h
                    self.lstm_outputs.append(cell_output)  
                    
        with tf.name_scope('lstm2'): 
            with tf.variable_scope('lstm2'):
                lstm_cell2 = rnn_cell_impl.TAMSRNN_CELL(self.lstm_hidden_size, self.period, forget_bias=1.0)
                init_state2 = lstm_cell2.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
                state = init_state2
                for time_step in range(self.n_steps):
                    if time_step > 0 : tf.get_variable_scope().reuse_variables() 	
                    (cell_h, state, _) = lstm_cell2(self.lstm_outputs[time_step], state, time_step)
                    cell_output = cell_h
                    self.lstm_outputs2.append(cell_output)
               
        self.lstm_output_final = tf.reshape(self.lstm_outputs2[-1], [-1, self.lstm_hidden_size])
         
        with tf.name_scope('softmax'):
            self.W_softm = weight_variable([self.lstm_hidden_size, self.num_class])
            self.B_softm = bias_variable([self.num_class])            
            
            self.prediction = tf.nn.softmax(tf.matmul(self.lstm_output_final, self.W_softm) + self.B_softm)
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_one_hot * tf.log(self.prediction + (1e-10) ), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        
        with tf.name_scope('accuracy'):
            correct_predict = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y_one_hot, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predict,"float"),name = "accuracy")
            
    
    
    
    
    
    









