from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import gensim
import math,random
import cPickle as cp

def load_data(filepath , length , dimension = 1):

    with open(filepath , 'rb') as fo:
        data , label = cp.load(fo)
        data = data
        label = label
        samples = data.shape[0]
        datas  = np.zeros((samples , length , dimension))
        labels = label.reshape(samples,)
        mi , ma = 60000 , 0
        for i in range(samples) :
            d_len = data[i].shape[0]
            mi = min(mi , d_len)
            ma = max(ma , d_len)
            if d_len >= length :
                data_temp = data[i][:length]
                data_mean = data_temp.mean(axis = 0)
                data_std = data_temp.std(axis = 0)
                datas[i] = (data_temp - data_mean) * 1.0 / data_std
            else :
                data_temp = data[i]
                data_mean = data_temp.mean(axis = 0)
                data_std = data_temp.std(axis = 0)
                datas[i][-d_len:] = (data_temp - data_mean) * 1.0 / data_std

    print('min_len : %d , max_len : %d ' %(mi , ma))
    return datas , labels


def next_batch_data(data, label, batch_size):
    while 1:
        epoch_size = int(math.ceil(data.shape[0] / batch_size))
        for i in range(epoch_size):
            if i < epoch_size - 1:
                data_batch = data[i * batch_size : (i+1) * batch_size]
                label_batch = label[i * batch_size : (i+1) * batch_size]
            else:
                data_batch = data[i * batch_size:]
                label_batch = label[i * batch_size:]
            yield data_batch, label_batch







