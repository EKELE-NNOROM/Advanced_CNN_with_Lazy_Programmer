#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 11:30:06 2018

@author: ekele
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

def get_normalized_data(input_directory):
    df = pd.read_csv(input_directory)
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:,1:].reshape(-1, 28, 28, 1) / 255.0
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 1,0)
    X = (X - mu) / std
    Y = data[:, 0].astype(np.int32)
    return X, Y

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    ind[np.arange(N), y] = 1
    return ind

X, Y = get_normalized_data('../large_files/fashionmnist/fashion-mnist_train.csv')


Y = y2indicator(Y)

batch_size = 32
epochs = 25

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(units = 128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 10, activation = 'softmax'))


###################################################################




#model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D())
#
#model.add(Conv2D(filters=64, kernel_size=(3, 3)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D())
#
#model.add(Conv2D(filters=128, kernel_size=(3, 3)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D())
#
#model.add(Flatten())
#model.add(Dense(units=300))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
#model.add(Dense(units=10))
#model.add(Activation('softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

r = model.fit(X, Y, batch_size=batch_size, epochs = epochs, \
              verbose = 1, validation_split = 0.33)
print('Returned:', r)
print(r.history.keys())


#Xtest, Ytest = get_normalized_data('../large_files/fashionmnist/fashion-mnist_test.csv')

df = pd.read_csv('../large_files/fashionmnist/fashion-mnist_test.csv')
data = df.values.astype(np.float32)
np.random.shuffle(data)
Xtest = data[:,1:].reshape(-1, 28, 28, 1) / 255.0
Ytest = data[:, 0].astype(np.int32)

Ytest = y2indicator(Ytest)

train_score = model.evaluate(X, Y, verbose=0)
test_score = model.evaluate(Xtest, Ytest, verbose=0)

print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])

print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
