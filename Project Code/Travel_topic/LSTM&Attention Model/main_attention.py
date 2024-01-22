# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:48:06 2023

@author: CS10 Group
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

plt.switch_backend('Agg')

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



outputfile_name = 'test_chart/'
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
X_train_test, X_test = np.squeeze(X[:337],axis = -1), np.squeeze(X[-1],axis = -1)
y_train_test, y_test = np.squeeze(y[:337],axis = -1), np.squeeze(y[-1],axis = -1)

train_size = int(len(X_train_test) * train_size)
X_train, X_val = X_train_test[:train_size], X_train_test[train_size + n_steps_in:len(X_train_test) - (n_steps_out - 1)]
y_train, y_val = y_train_test[:train_size], y_train_test[train_size + n_steps_in:len(X_train_test) - (n_steps_out - 1)]


shuffle_idx = np.random.permutation(np.arange(len(X_train)))

print("train data shape: ",X_train.shape)

num_data_per_batch = 36
num_batch = int(train_size/num_data_per_batch) + 1

#X_val = np.expand_dims(X_val, -1)

X_val = torch.FloatTensor(X_val)
X_val = X_val.cuda()


#y_val = np.expand_dims(y_val, -1)
y_val = torch.FloatTensor(y_val)
y_val = y_val.cuda()

val = y_val.cpu().detach().numpy()
val = sc.inverse_transform(val)

'''
print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)
print("X_val: ",X_val.shape)
print("y_val: ",y_val.shape)
'''



num_embbading = 128
size_kernel = 3
lr = 1e-5  # learning rate
epochs = 2000
loss_function = torch.nn.MSELoss()
model_save_path = 'ConvTransformer_nologsparse.pth'

model = TransformerTimeSeries(n_steps_in,n_steps_out,num_embbading,size_kernel)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_epoch_loss = []
eval_epoch_loss = []

acc_best = -10000000
epoch_best = 0
best_y = [] 

for e, epoch in enumerate(range(epochs)):
    batched_x = np.zeros([num_batch,num_data_per_batch,n_steps_in])
    batched_y = np.zeros([num_batch,num_data_per_batch,n_steps_out])
    for i in range(num_batch):
        stra_idx = num_data_per_batch*i
        end_idx = num_data_per_batch*(i+1)
        if end_idx > len(shuffle_idx):
            tmp_x = X_train[shuffle_idx[stra_idx:],:]
            tmp_y = y_train[shuffle_idx[stra_idx:],:]
        else:
            tmp_x = X_train[shuffle_idx[stra_idx:end_idx],:]
            tmp_y = y_train[shuffle_idx[stra_idx:end_idx],:]
            
        batched_x[i,0:len(tmp_x),:] = tmp_x
        batched_y[i,0:len(tmp_y),:] = tmp_y
    
    loss_list = []
    eval_loss_list = []
    for idx in range(batched_x.shape[0]):
        X_train_batch = batched_x[idx,:,:]
        y_train_batch = batched_y[idx,:,:]
        
        X_train_batch = torch.FloatTensor(X_train_batch)
        X_train_batch = X_train_batch.cuda()
        
        y_train_batch = torch.FloatTensor(y_train_batch)
        y_train_batch = y_train_batch.cuda()
        
        y_pred = model(X_train_batch)
        
        #print("y_pred: ",y_pred.shape,"\ny_train_batch: ",y_train_batch.shape)
        loss = loss_function(y_pred,y_train_batch)
        
        optimizer.zero_grad()
    
        loss.backward()
    
        optimizer.step()
        
        model.eval()
        y_val_predicted = model(X_val)
        model.train()
        
        y_val = y_val.squeeze()
        eval_loss = loss_function(y_val_predicted,y_val)
        
        val_pred = y_val_predicted.cpu().detach().numpy()
        #val_pred = np.expand_dims(val_pred, -1)
        val_pred = sc.inverse_transform(val_pred)
        
        acc = 1 - np.mean(np.absolute(val_pred - val)/val)
        
        if acc_best < acc:
            acc_best = acc
            epoch_best = epoch
            best_y = val_pred
            if epoch > epochs/10:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_loss,
                }, model_save_path)
        if epoch%100 == 0:
            loss_list.append(torch.mean(loss))
            eval_loss_list.append(torch.mean(eval_loss))

    if epoch%100 == 0:
        loss_list = torch.FloatTensor(loss_list)
        eval_loss_list = torch.FloatTensor(eval_loss_list)
        print("Epoch {}: Train loss: {} \t Validation loss: {} \t MAPE={}".format(e,
                                                                                 torch.mean(loss_list),
                                                                                 torch.mean(eval_loss_list), acc))
    
        print("MAPE best={}".format(acc_best))
    
    if (epoch+1)%500 == 0:
        
        y_test = []

        y_test_predicted = []
        
        y_test_predicted_all = val_pred
        
        
        idx = 0
        while idx < len(y_test_predicted_all):
            y_test_predicted = np.concatenate((y_test_predicted,y_test_predicted_all[idx]))
            y_test = np.concatenate((y_test,val[idx]))
            idx = idx + n_steps_out
        
        #print("y_test_predicted: ",y_test_predicted)
        #print("y_test: ",y_test)
        
        '''
        # Plot the line chart
        ax.plot(val_pred,label = "best predict",color = 'r')
        ax.plot(val,label = "groud true",color = 'k')
        
        # Set the x-axis label
        ax.set_xlabel('Month')
        
        # Set the y-axis label
        ax.set_ylabel('Amount of Tourists')
        
        # Set the chart title
        ax.set_title('prediction of tourist data\nacc: ' +str(acc))
        
        # save the chart
        plt.savefig(outputfile_name + 'reuslt_of_training_'  + str(epoch+1) + '.png')

        plt.close()
        
        '''
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
        plt.savefig(outputfile_name + 'reuslt_of_training_'  + str(epoch+1) + '.png')
        
        plt.close()
        

if len(best_y) > 0:
    fig, ax = plt.subplots()
    
    y_test = []

    y_test_predicted = []
    
    y_test_predicted_all = best_y
    
    
    idx = 0
    while idx < len(y_test_predicted_all):
        y_test_predicted = np.concatenate((y_test_predicted,y_test_predicted_all[idx]))
        y_test = np.concatenate((y_test,val[idx]))
        idx = idx + n_steps_out
    
    
    # Plot the line chart
    ax.plot(y_test_predicted,label = "best predict",color = 'r')
    ax.plot(y_test,label = "groud true",color = 'k')
    
    # Set the x-axis label
    ax.set_xlabel('Month')
    
    # Set the y-axis label
    ax.set_ylabel('Amount of Tourists')
    
    # Set the chart title
    ax.set_title('prediction of tourist data\nMAPE: ' +str(acc_best))
    
    # Show the chart
    plt.savefig(outputfile_name + 'best.png')
    
    plt.close()

'''
X_test = torch.FloatTensor(X_test)
X_test = X_test.cuda()

model_para = torch.load(model_save_path)
model.load_state_dict(model_para['model_state_dict'])

y_test_predicted = model(X_test)

test_pred = y_test_predicted.cpu().detach().numpy()
test_pred = sc.inverse_transform(test_pred)

test = sc.inverse_transform(y_test)

acc = 1 - np.mean(np.absolute(test_pred - test)/test)

print("acc on covid period: ",acc)

plt.close()
fig, ax = plt.subplots()

# Plot the line chart
ax.plot(test_pred[0,:],label = "best predict last",color = 'r')
ax.plot(test[0,:],label = "groud true",color = 'k')

# Set the x-axis label
ax.set_xlabel('Index')

# Set the y-axis label
ax.set_ylabel('Value')

# Set the chart title
ax.set_title('prediction of tourist data')

# Show the chart
plt.savefig(outputfile_name + 'best_first_covid.png')
'''












