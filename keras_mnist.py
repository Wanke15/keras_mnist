#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:21:54 2016

@author: farmer
"""

import os
import gzip
import cPickle as pickle
from urllib import urlretrieve
import numpy as np

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input,Dense
from keras.utils import np_utils


batch_size = 128
num_epochs = 40
hidden_size = 640


num_train = 50000
num_test = 10000


height,width,depth = 28,28,1
num_classes = 10

'''
(X_train,y_train),(X_test,y_test) = mnist.load_data()


X_train = X_train.reshape(num_train,height*width)
X_test = X_test.reshape(num_test,height*width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train,num_classes)
Y_test = np_utils.to_categorical(y_test,num_classes)
'''


def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
        print "下载完毕!"
    else:
        print "文件已存在!"
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    return X_train, y_train, X_val, y_val, X_test, y_test
    
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
print type(y_train),type(y_val)
print "数据导入成功!"

#print np.shape(X_train)
#X_train = X_train.reshape(num_train,height*width)
#X_test = X_test.reshape(num_test,height*width)
X_train = np.vstack((X_train,X_val))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_val = np_utils.to_categorical(y_val,num_classes)
Y_train = np_utils.to_categorical(y_train,num_classes)
Y_train = np.vstack((Y_train,Y_val))
Y_test = np_utils.to_categorical(y_test,num_classes)
print "Transformation confirmed!"


inp = Input(shape = (height*width,))
hidden_1 = Dense(hidden_size,activation = 'relu')(inp)
hidden_2 = Dense(hidden_size,activation = 'relu')(hidden_1)
hidden_3 = Dense(hidden_size,cativation = 'relu')(hidden_2)
out = Dense(num_classes,activation = 'softmax')(hidden_2)


model = Model(input = inp,output = out)


model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


model.fit(X_train,Y_train,
          batch_size = batch_size,nb_epoch = num_epochs,
          verbose = 1,validation_split = 0.1)
model.evaluate(X_test,Y_test,verbose = 1)
