# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 02:12:21 2019

@author: raymond
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:43:59 2019

@author: raymond
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

def get_all_dataset_dir():
    preprocess_stock_data_DIRs = []
    cwd = os.getcwd()
    stock_folder = os.path.join(cwd, 'stock_dataset_v3','conbine_stock_and_tweet')
    preprocess_stock_data_DIRs = [os.path.join(stock_folder, f) for f in os.listdir(stock_folder)]
    return preprocess_stock_data_DIRs


def read_all_dataset_dataframe(dataset_DIRs):
    all_dataset = []
    for dataset_DIR in dataset_DIRs:
        data = pd.read_csv(dataset_DIR) 
        all_dataset.append(data)
    
    return all_dataset


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)



def buildTimesetpData(data, pastDay, futureDay):
    X, Y = [], []
    for i in range(data.shape[0]-futureDay-pastDay):
        X.append(np.array(data[i:i+pastDay]))
        Y.append(np.array(data[i+pastDay:i+pastDay+futureDay,0]))
    return np.array(X), np.array(Y)

def shuffle(X,Y):
    
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(data,split):
    X_train = data[:-split]
    X_val = data[-split:]
    return X_train, X_val




def buildManyToOneModel_selection(shape,futherday,stack_num,nodes):
  model = Sequential()
  
  if stack_num > 0:
      model.add(LSTM(nodes, input_length=shape[1], input_dim=shape[2], return_sequences=True))
      model.add(Dropout(0.2))
  else:
      model.add(LSTM(nodes, input_length=shape[1], input_dim=shape[2]))
      model.add(Dropout(0.2))
  
  for i in range(stack_num-1):
      model.add(LSTM(nodes, return_sequences=True))
      model.add(Dropout(0.2))
      
  if stack_num > 0:
      model.add(LSTM(nodes))
      model.add(Dropout(0.2))
  
  model.add(Dense(futherday))
  model.compile(loss='mse', optimizer="adam",metrics=['mse'])
  model.summary()
  return model


def plot_model_loss(model_history_list,fig_name,legend_names):
    

    
    plt.figure(fig_name + " train") 
    #plt.subplot(211)
    for model_history in model_history_list:
        plt.plot(model_history.history['mse'])
    plt.grid(True)
    plt.ylim(0.001, 0.06)
    plt.title('train_MSE_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend_names, loc='upper right')    
    
    plt.figure(fig_name+ " val") 
    #plt.subplot(212)
    for model_history in model_history_list:
        plt.plot(model_history.history['val_mse'])
    plt.grid(True)
    plt.ylim(0.001, 0.06)
    plt.title('val_MSE_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legend_names, loc='upper right')         



def check(X,Y,futherday):
    temp_X = []
    temp_Y = []
    for i,j in zip(X,Y):
        if j.shape[0] == futherday:
            temp_X.append(i)
            temp_Y.append(j)
            
    return np.array(temp_X),np.array(temp_Y)


if __name__ == "__main__":    
    stock_num = 1
    features_set_list = [['Adj Close'],
                         ['Adj Close','Sentiment Scores'],
                         ['Adj Close','Sentiment Scores','Change in %'],
                         ['Adj Close','Sentiment Scores','Change in %', 'year', 'month', 'date','day']]
    
    window_sizes = [10,15,20,25,30]
    stack_nums = [0,1,2,3]
    nodes = [32,64,128,256]
    model_history_list = []
    
    for window_size in window_sizes:
        
        # Read training Data
        tf.random.set_seed(13)
        all_dataset_dir = get_all_dataset_dir()
        all_dataset = read_all_dataset_dataframe(all_dataset_dir)
        df = all_dataset[stock_num].copy()
        
        
        
        df_obj = pd.DataFrame(columns = features_set_list[1])
        for f in features_set_list[1]:
            df_obj[f] = df[f]
        
        
        
        #splitData
        split_for_val = 72
        train, val = splitData(df_obj.copy(), split_for_val)
        
        
        #Normalization
        scaler = MinMaxScaler(feature_range=(-1, 1))
        #scaler = StandardScaler()
        scaler.fit(df_obj)
        train_norm = scaler.transform(train)
        val_norm = scaler.transform(val)
        
        Y = np.array(df_obj.iloc[:]["Adj Close"])[:,np.newaxis]
        #Y_scaler = StandardScaler()
        Y_scaler = MinMaxScaler(feature_range=(-1, 1))
        Y_scaler.fit(Y)
        
        
        #buildTimesetpData
        pastdays = window_size
        futherday = 1
        STEP = 1
        
        X_train_norm, Y_train_norm = multivariate_data(train_norm, train_norm[:, 0], 0,
                                                           len(train_norm), pastdays,
                                                           futherday, STEP)
        X_val_norm, Y_val_norm = multivariate_data(val_norm, val_norm[:, 0],
                                                       0, len(val_norm), pastdays,
                                                       futherday, STEP)
        
        
        X_train_norm,Y_train_norm = check(X_train_norm,Y_train_norm,futherday)
        X_val_norm,Y_val_norm = check(X_val_norm,Y_val_norm,futherday)
        
        
        X_train_norm, Y_train_norm = shuffle(X_train_norm, Y_train_norm)
        X_val_norm, Y_val_norm = shuffle(X_val_norm, Y_val_norm)
        
        
        
        #config LSTM model
        model = buildManyToOneModel_selection(X_train_norm.shape,futherday,0, 32)
        history  = model.fit(X_train_norm, Y_train_norm, epochs=400, batch_size=8, validation_data=(X_val_norm, Y_val_norm))
        model_history_list.append(history)
        
        
    
    fig_name = "window size selection"
    legend_names=["window size = 10","window size = 15","window size = 20","window size = 25","window size = 30"]
    plot_model_loss(model_history_list,fig_name,legend_names)
    
#    fig_name = "feature selection"
#    legend_names=["AdjClose","AdjClose + Sentiment","AdjClose + Sentiment + Change in %","AdjClose + Sentiment + Change in % + Date"]
#    plot_model_loss(model_history_list,fig_name,legend_names)
#       
#    fig_name = "model architecture - nodes of layer selection"
#    legend_names=["LSTM 32","LSTM 64","LSTM 128","LSTM 256"]
#    plot_model_loss(model_history_list,fig_name,legend_names)
#     
#    fig_name = "model architecture - layer stack selection"
#    legend_names=["LSTM 32","LSTM 32 + LSTM 32","LSTM 32 + LSTM 32 + LSTM 32","LSTM 32 + LSTM 32 + LSTM 32 + LSTM 32"]
#    plot_model_loss(model_history_list,fig_name,legend_names)
    


