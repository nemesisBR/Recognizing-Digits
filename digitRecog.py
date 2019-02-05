# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras


raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
labels = raw_data.iloc[:,0:1].values
num = raw_data.iloc[:,0:1].values
pixel_values = raw_data.iloc[:,1:785].values

labels = keras.utils.to_categorical(labels, num_classes = 10)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D

model = Sequential()

model.add(Dense(100,kernel_initializer='uniform',activation='sigmoid',input_dim = 784))


model.add(Dense(100,activation='sigmoid',kernel_initializer='uniform'))


model.add(Dense(100,activation='sigmoid',kernel_initializer='uniform'))


model.add(Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

model.fit(pixel_values, labels,epochs=25,batch_size=20)

label_pred = model.predict(test_data)

final = []
label_pred = np.array(label_pred)
for p in range(len(label_pred)):
    final.append(label_pred[p].tolist().index(max(label_pred[p])))
    
count = 0
final = np.array(final)       

for x in range(len(final)):
    if final[x] == num[x]:
        count = count + 1
