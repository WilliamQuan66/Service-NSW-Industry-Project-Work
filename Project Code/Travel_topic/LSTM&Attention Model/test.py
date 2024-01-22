# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:37:18 2023

@author: CS10 Group
"""

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

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

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,k=5):
        super(context_embedding,self).__init__()
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)

    def forward(self,x):
        x = self.causal_convolution(x)
        return torch.tanh(x)

# model class
class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper
    
    context_embedding parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel
        
    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)
    
    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector
    
    """
    def __init__(self,num_step_in,num_step_out,num_embbading,num_kernel):
        super(TransformerTimeSeries,self).__init__()
        self.input_embedding = context_embedding(1,num_embbading,num_kernel)
        self.positional_embedding = torch.nn.Embedding(num_embbading*2,num_embbading)

        
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model = num_embbading,nhead = 8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers = 2)
        
        self.fc1 = torch.nn.Linear(num_embbading,1)
        self.fc2 = torch.nn.Linear(num_step_in,num_step_out)
        '''
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_embbading,      
                out_channels=num_embbading,     
                kernel_size=(num_step_out+1,1),  
                stride= 1,           
                padding= 0,      
            ),      
            nn.ReLU(),
            )
        '''
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_step_in,      
                out_channels=num_step_in,     
                kernel_size=(num_embbading,1),  
                stride= 1,           
                padding= 0,      
            ),      
            nn.ReLU(),
            )
        '''
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=round(num_embbading/2),      
                out_channels=1,     
                kernel_size=1,  
                stride= 1,           
                padding= 0,      
            ),      
            nn.ReLU(),
            )
        '''
        #self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self,x):
        
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        #print("in x: ",x.shape)
        z = x.unsqueeze(1)
        #print("uq x:",z.shape)

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2,0,1)
        
        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1,0,2)
        
        input_embedding = z_embedding + positional_embeddings
        
        transformer_embedding = self.transformer_decoder(input_embedding)
        
        output = transformer_embedding.permute(1,0,2)
        #print("output shape: ",output.shape)
        output = output.unsqueeze(-1)
        #print("output shape: ",output.shape)
        output = self.conv1(output)
        #print("output shape: ",output.shape)
        #output = self.conv2(output)
        #output = output.permute(0,2,1,3)
        
        output = output.squeeze()
        #print("output shape: ",output.shape)
        output = self.fc2(output)
        output = output.squeeze()
        
        return output

outputfile_name = 'test_output/'
if not os.path.isdir(outputfile_name):
    os.mkdir(outputfile_name)

model_save_path = 'ConvTransformer_nologsparse.pth'
data = pd.read_excel('travel_one_more.xlsx')
train_data = data.iloc[:, 0:1].values
sc = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = sc.fit_transform(train_data)
train_data_scaled = train_data_scaled

n_steps_in = 12
n_steps_out = 6

num_embbading = 128
size_kernel = 3

model = TransformerTimeSeries(n_steps_in,n_steps_out,num_embbading,size_kernel)
model = model.cuda()

model_para = torch.load(model_save_path)
model.load_state_dict(model_para['model_state_dict'])
model.eval()
'''
#here
num_iteration = 24

X,_ = to_supervised(train_data_scaled, n_steps_in, n_steps_out)

#before 2020
X_train_test = X[:337]
last_ordinary = X_train_test[-1]
last_ordinary = sc.inverse_transform(last_ordinary)
average_ordinary = int(np.mean(last_ordinary))

x_feature = train_data_scaled[-12:].T

x_feature_show = sc.inverse_transform(x_feature)
print("x_feature: ",x_feature_show)

x_feature = torch.FloatTensor(x_feature)
x_feature = x_feature.cuda()

x_feature_now = x_feature
y_test_predicted = torch.Tensor().cuda()
for i in range(num_iteration):
    y_test_predicted_now = model(x_feature_now)
    y_test_predicted = torch.cat((y_test_predicted,y_test_predicted_now))
    x_feature_now = x_feature_now.squeeze()
    x_feature_now = torch.cat((x_feature_now[-6:],y_test_predicted_now))
    x_feature_now = x_feature_now.unsqueeze(0)

test_pred = y_test_predicted.cpu().detach().numpy()
test_pred = np.expand_dims(test_pred, -1)
test_pred = sc.inverse_transform(test_pred)

np.savetxt(outputfile_name + 'test_pred.csv', test_pred, delimiter=',')
print("saved to csv")

readched_idx = -1
for i,pred in enumerate(test_pred):
    if not(pred < average_ordinary):
        readched_idx = i
        break

start_date = datetime(2023, 4, 1)

dates = []
num_month_show = 6

current_date = start_date
for i in range(round(num_iteration*n_steps_out/num_month_show)):
    dates.append(current_date)
    current_date += relativedelta(months=num_month_show)

x_axis = [date.strftime('%Y-%m') for date in dates]

fig, ax = plt.subplots(figsize=(16, 16))

# Plot the line chart
ax.plot(test_pred,label = "prediction",color = 'r')
ax.legend()

# Average last ordinary year
ax.axhline(y=average_ordinary, color='blue',linestyle='--')
text_x = 20
text_y = average_ordinary*1.01
text = 'Average touriest arrival in 2020: {}'.format(average_ordinary)
ax.text(text_x, text_y, text, ha='center')

if not(readched_idx == -1):
    
    ax.axvline(x=readched_idx, color='green', linestyle=':',ymax=(test_pred[readched_idx] - ax.get_ylim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]))
    text_x2 = readched_idx
    text_y2 = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.1
    text2 = x_axis[round(readched_idx/num_month_show)]
    ax.text(text_x2, text_y2, text2, ha='center')

# Set the x-axis label
ax.set_xlabel('Month')
ax.set_xticks(range(0,len(x_axis)*num_month_show,num_month_show))
ax.set_xticklabels(x_axis,rotation=90)
ax.tick_params(axis='x', labelsize=12)

# Set the y-axis label
ax.set_ylabel('Amount of Tourists Arrived')

# Set the chart title
ax.set_title('Prediction of short-term tourists arrived in NSW each month')

plt.savefig(outputfile_name + 'feature_prediction_' + str(len(y_test_predicted)) + '_month.png',dpi=500)
print("saved to png")


'''
X, y = to_supervised(train_data_scaled, n_steps_in, n_steps_out)

#before 2020
X_train_test, X_test = np.squeeze(X[:337],axis = -1), X[-1]
y_train_test, y_test = np.squeeze(y[:337],axis = -1), np.squeeze(y[-1],axis = -1)

train_size = int(len(X_train_test) * 0.8)
X_train, X_val = X_train_test[:train_size], X_train_test[train_size + n_steps_in:len(X_train_test) - (n_steps_out - 1)]
y_train, y_val = y_train_test[:train_size], y_train_test[train_size + n_steps_in:len(X_train_test) - (n_steps_out - 1)]



X_test = X_val
y_test = []

X_test = torch.FloatTensor(X_test)
X_test = X_test.cuda()


#y_test_predicted = model(X_test)


y_test_predicted = torch.Tensor().cuda()

y_test_predicted_all = model(X_test)


idx = 0
while idx < len(y_test_predicted_all):
    y_test_predicted = torch.cat((y_test_predicted,y_test_predicted_all[idx]))
    y_test = np.concatenate((y_test,y_val[idx]))
    idx = idx + n_steps_out



test_pred = y_test_predicted.cpu().detach().numpy()
test_pred = np.expand_dims(test_pred, -1)
test_pred = sc.inverse_transform(test_pred)

y_test = np.expand_dims(y_test, -1)
test = sc.inverse_transform(y_test)

acc = 1 - np.mean(np.absolute(test_pred - test)/test)

print("MAPE: ",acc)

np.savetxt(outputfile_name + 'test_pred.csv', test_pred, delimiter=',')
print("saved to csv")

start_date = datetime(2015, 7, 1)

end_date = datetime(2019, 12, 1)

dates = []

current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += relativedelta(months=1)

x_axis = [date.strftime('%Y-%m') for date in dates]

fig, ax = plt.subplots(figsize=(16, 16))

# Plot the line chart
ax.plot(test_pred,label = "prediction",color = 'r')
ax.plot(test,label = "groud true",color = 'k')
ax.legend()

# Set the x-axis label
ax.set_xlabel('Month')
ax.set_xticks(range(0,len(x_axis)))
ax.set_xticklabels(x_axis,rotation=90)
ax.tick_params(axis='x', labelsize=12)

# Set the y-axis label
ax.set_ylabel('Amount of Tourists Arrived')

# Set the chart title
ax.set_title('Prediction of short-term tourists arrived in NSW each month\nMAPE: ' +str(acc))

plt.savefig(outputfile_name + 'test_result_' + str(len(y_test_predicted)) + '_month.png',dpi=500)
print("saved to png")
#plt.savefig('all_output/covid_test_result_' + str(num_squeeze_size) + '_times_squeeze.png')











