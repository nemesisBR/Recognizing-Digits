# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
labels = raw_data.iloc[:,0:1].values
pixel_values = raw_data.iloc[:,1:785].values

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D

model = Sequential()

model.add(Dense(units=16,kernel_initializer='uniform',activation='relu',input_dim = 784))

model.add(Dense(units=16,activation='relu',init='uniform'))

model.add(Dense(units=1,activation='relu',init='uniform'))

model.compile(loss = 'mean_squared_error', optimizer = 'sgd')

model.fit(pixel_values, labels,epochs=50,batch_size=20)

label_pred = model.predict(test_data)


