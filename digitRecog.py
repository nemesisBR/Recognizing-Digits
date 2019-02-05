# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_data = np.array(test_data)
labels = raw_data.iloc[:,0:1].values
num = raw_data.iloc[:,0:1].values
pixel_values = raw_data.iloc[:,1:785].values
pixel_values = pixel_values.reshape(pixel_values.shape[0],28,28)
pixel_values = keras.utils.normalize(pixel_values, axis = 1)

test_data = test_data.reshape(test_data.shape[0],28,28)
test_data = keras.utils.normalize(test_data, axis = 1)


labels = keras.utils.to_categorical(labels, num_classes = 10)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D

model = Sequential()

model.add(Flatten())
model.add(Dense(100,kernel_initializer='uniform',activation='sigmoid',input_dim = 784))


model.add(Dense(100,activation='sigmoid',kernel_initializer='uniform'))


model.add(Dense(100,activation='sigmoid',kernel_initializer='uniform'))


model.add(Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])

model.fit(pixel_values, labels,epochs=25,batch_size=50)

label_pred = model.predict(test_data)

final = []
label_pred = np.array(label_pred)
for p in range(len(label_pred)):
    final.append(label_pred[p].tolist().index(max(label_pred[p])))
    
final = pd.Series(final)
imageId = pd.Series([x for x in range(1,28001)])

result = pd.concat([imageId, final], axis =1, names = ['ImageId', 'Label'])

result.to_csv('submission.csv', index=False )