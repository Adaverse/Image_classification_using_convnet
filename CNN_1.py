#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:18:03 2018

@author: vicky
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import tensorflow
from PIL import Image
from numpy import *

#sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

#keras 
from keras.utils import np_utils
from keras.models import Model  #for sequential model
from keras.layers import AveragePooling2D, MaxPooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras import Input

path1 = "/Users/vicky/Desktop/data/input_data"
path2 = "/Users/vicky/Desktop/data/input_data_resized"

listing = sorted(os.listdir(path1))
num_samples=size(listing)
print(num_samples)

img_rows = 128
img_cols = 128

for file in listing:
    im = Image.open(path1 + '//' +file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(path2 + '//' + file,'JPEG')
    

imlist = sorted(os.listdir(path2))

img1 = array(Image.open(path2 + '/' + imlist[0]))
m,n = img1.shape[0:2]
imnbr = len(imlist)

num_samples=size(imlist)

immatrix = array([array(Image.open(path2 + '/' + im2)).flatten()
            for im2 in imlist],'f')
    
label = np.ones((num_samples,),dtype = int)
label[0:202]=0
label[202:404]=1
label[404:606]=2
label[606:808]=3

data,label = shuffle(immatrix,label, random_state=2)
train_data=[data,label]

img = immatrix[150].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img,cmap='gray')

#sanity check
print (train_data[0].shape)
print (train_data[1].shape)


batch_size = 8
nb_classes = 4
np_epoch = 20
img_channels = 1
nb_filters = 32
nb_pool = 2
nb_conv = 3

(X,y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print ('X_train_shape:', X_train.shape)
print ('X_test_shape:', X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i=5
plt.imshow(X_train[i,0], interpolation='nearest')
print ('label: ', Y_train[i,:])

def model(input_shape):
    X_input = Input(input_shape)
    
    #layer1
    X = Conv2D(6,(5,5),strides = (1,1))(X_input)
    X = BatchNormalization(axis = 3,)(X)
    X = activation('relu')(X)
    X = AveragePooling2D((2,2))(X)
    
    #layer2
    X = Conv2D(6,(5,5),strides = (1,1))(X)
    X = BatchNormalization(axis = 3,)(X)
    X = activation('relu')(X)
    X = MaxPooling2D((1,1))(X)
    
    #layer3
    X = Conv2D(6,(2,2),strides = (4,4))(X)
    X = BatchNormalization(axis = 3,)(X)
    X = activation('relu')(X)
    X = MaxPooling2D((4,4))(X)
    
    #layer4
    X = Flatten(X)
    X = Dense(96,activation='sigmoid')(X)
    X = Dense(4,activation = 'softmax')(X)
    
    cnn_model = Model(inputs = X_input, outputs = X)
    
    return cnn_model

    
model1 = model((128,128,1))
