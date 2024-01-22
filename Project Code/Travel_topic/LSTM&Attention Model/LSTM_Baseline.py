# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:48:06 2023

@author: CS10 Group
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import math
import sklearn.metrics as skm
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import warnings
from datetime import datetime
 
warnings.filterwarnings('ignore')

#plt.switch_backend('Agg')

def to_supervised(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def lstm_model(train_x, train_y, verbose_set=0, epochs_num=200, batch_size_set=4):
    '''
    Encoder-Decoder LSTM model
    '''
    
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    model.fit(train_x, train_y, epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set)
    return model



outputfile_name = 'Baseline_test_chart/'
if not os.path.isdir(outputfile_name):
    os.mkdir(outputfile_name)


data = pd.read_excel('0421_travel.xlsx')
train_data = data.iloc[:, 0:1].values
sc = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = sc.fit_transform(train_data)
train_data_scaled = train_data_scaled

n_steps_in = 12
n_steps_out = 6
train_size = 0.8

X, y = to_supervised(train_data_scaled, n_steps_in, n_steps_out)

#before 2020
X_train_test, X_test = X[:337], X[-1]
y_train_test, y_test = y[:337],y[-1]

train_size = int(len(X_train_test) * train_size)
X_train, X_val = X_train_test[:train_size], X_train_test[train_size + n_steps_in:len(X_train_test) - (n_steps_out - 1)]
y_train, y_val = y_train_test[:train_size], y_train_test[train_size + n_steps_in:len(X_train_test) - (n_steps_out - 1)]

print("train data shape: ",X_train.shape)
print("train label shape: ",y_train.shape)
print("test data shape: ",X_val.shape)
print("test label shape: ",y_val.shape)

val = sc.inverse_transform(np.squeeze(y_val,axis=-1))




model = lstm_model(X_train, y_train, verbose_set=0, epochs_num=3000, batch_size_set=4)

y_val_predicted = model.predict(X_val, verbose=0)
y_val_predicted = y_val_predicted
y_val_predicted = np.squeeze(y_val_predicted,axis=-1)

val_pred = sc.inverse_transform(y_val_predicted)
acc = 1 - np.mean(np.absolute(val_pred - val)/val)

print("Baseline Testing MAPE: ",acc)

y_test = []

y_test_predicted = []

y_test_predicted_all = val_pred


idx = 0
while idx < len(y_test_predicted_all):
    y_test_predicted = np.concatenate((y_test_predicted,y_test_predicted_all[idx]))
    y_test = np.concatenate((y_test,val[idx]))
    idx = idx + n_steps_out

fig, ax = plt.subplots()
ax.plot(y_test_predicted,label = "best predict",color = 'r')
ax.plot(y_test,label = "groud true",color = 'k')

# Set the x-axis label
ax.set_xlabel('Month')

# Set the y-axis label
ax.set_ylabel('Amount of Tourists')

# Set the chart title
ax.set_title('prediction of tourist data\nMAPE: ' +str(acc))

# save the chart
plt.savefig(outputfile_name + 'baseline_result.png')

plt.close()

np.savetxt(outputfile_name + 'test_pred.csv', y_test_predicted, delimiter=',')
print("saved to csv")













