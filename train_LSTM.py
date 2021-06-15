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
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from sklearn.metrics import mean_squared_error 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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


def shuffle(X,Y):
    
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(data,split):
    X_train = data[:-split]
    X_val = data[-split:]
    return X_train, X_val

def buildManyToOneModel(shape,futherday):
  model = Sequential()

  model.add(LSTM(32,input_length=shape[1], input_dim=shape[2]))
  model.add(Dropout(0.2))
  model.add(Dense(futherday))
  model.compile(loss='mse', optimizer="adam",metrics=['mse'])
  model.summary()
  return model

def plot_model_loss(history, window_name = "Model Loss"):
    
    plt.figure(window_name)
    
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('MSE loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')


def model_performance_visualization(train_norm, val_norm, ref_train, ref_val, pastdays, futherday, model, window_name):

    mse_train = []
    mse_val=[]
    plt.figure(window_name)
    STEP = 1
    
    X_train_norm, Y_train_norm = multivariate_data(train_norm, train_norm[:, 0], 0,
                                                       len(train_norm), pastdays,
                                                       futherday, STEP)
    X_val_norm, Y_val_norm = multivariate_data(val_norm, val_norm[:, 0],
                                                   0, len(val_norm), pastdays,
                                                   futherday, STEP)
    
    
    X_train_norm,Y_train_norm = check(X_train_norm,Y_train_norm,futherday)
    X_val_norm,Y_val_norm = check(X_val_norm,Y_val_norm,futherday)
    
    
    pred_train_norm = model.predict(X_train_norm)
    pred_train = Y_scaler.inverse_transform(pred_train_norm)
    
    
    pred_val_norm = model.predict(X_val_norm)
    pred_val = Y_scaler.inverse_transform(pred_val_norm)
    
    
    plt.subplot(311)
    x_axis_train = list(range(0,len(pred_train)))
    plt.plot(x_axis_train,pred_train,x_axis_train,ref_train)
    plt.legend(('Pred_train', 'Ref_train'),loc='upper right')
    
    plt.subplot(312)
    x_axis_val = list(range(0,len(ref_val)))
    plt.plot(x_axis_val,pred_val,x_axis_val,ref_val)
    plt.legend(('Pred_val', 'Ref_val'),loc='upper right')
    
    plt.subplot(313)
    extend_axis = list(range(len(pred_train),len(ref_val)+len(pred_train)))
    plt.plot(x_axis_train,100*(pred_train-ref_train)/abs(ref_train))
    plt.plot(extend_axis,100*(pred_val-ref_val)/abs(ref_val))
    plt.legend(('train_error', 'val_error'),loc='upper right')
    plt.show()
    
    mse_train = mean_squared_error(pred_train, ref_train)
    mse_val = mean_squared_error(pred_val, ref_val)
    
    return mse_train, mse_val

def check(X,Y,futherday):
    temp_X = []
    temp_Y = []
    for i,j in zip(X,Y):
        if j.shape[0] == futherday:
            temp_X.append(i)
            temp_Y.append(j)
            
    return np.array(temp_X),np.array(temp_Y)


    
def get_tgt_tweet_dir():
    tweet_data_DIRs = []
    cwd = os.getcwd()
    tweet_folder = os.path.join(cwd, 'tweet_test',"sentiment_analysis_test")
    tweet_data_DIRs = [os.path.join(tweet_folder, f) for f in os.listdir(tweet_folder)]
    return tweet_data_DIRs


def read_tgt_tweet_dataframe(dataset_DIRs):
    all_dataset = []
    for dataset_DIR in dataset_DIRs:
        data = pd.read_csv(dataset_DIR) 
        all_dataset.append(data)
    
    return all_dataset




def test_inference(model,tgt_tweet_df,df_obj,X_scaler,Y_scaler,pastdays):
    
    tgt_date = tgt_tweet_df["Date"].values
    
    
    if "Sentiment Scores" not in df_obj.keys():
        
        offset = 2
        X_start = df_obj["Adj Close"][df_obj.index[range(-pastdays-offset,-offset)]].values.reshape(-1,1)
        X_start = X_scaler.transform(X_start)
        
        pred = X_start
        preds = []
        for i in range(len(tgt_date)):
            pred = model.predict(pred.reshape((1,pred.shape[0],pred.shape[1])))
            preds.append(pred.squeeze())
            
    else:
        offset = 0
        X_start = df_obj["Adj Close"][df_obj.index[range(-pastdays-offset,-offset)]].values.reshape(-1,1)
        X_tweet = df_obj["Sentiment Scores"][df_obj.index[range(-pastdays-offset,-offset)]].values.reshape(-1,1)
        X_start = np.hstack((X_start, X_tweet))
        X_start = X_scaler.transform(X_start)
        
        
        tgt_sentiment = tgt_tweet_df["Sentiment Scores"].values
        tgt_sentiment = tgt_sentiment.reshape(-1,1)
        pad = np.zeros(tgt_sentiment.shape[0]).reshape(-1,1)
        tgt_sentiment = np.concatenate((pad,tgt_sentiment), axis=1)
        tgt_sentiment= X_scaler.transform(tgt_sentiment)[:,1]
        
        
        X = X_start
        preds = []
        for sentiment in tgt_sentiment:
            
            pred = model.predict(X.reshape((1,X.shape[0],X.shape[1])))
            preds.append(pred.squeeze())
            pred_with_tweet = np.hstack((pred,sentiment.reshape(-1,1)))
            X = np.vstack((X[1:],pred_with_tweet))
            
            
    preds = np.array(preds)
    preds = Y_scaler.inverse_transform(preds.reshape(1,-1))
    
    preds_test_df = pd.DataFrame({'Date':tgt_date,'pred':preds.reshape(-1)})
    preds_test_df = preds_test_df[preds_test_df.Date != "2015-12-25"]
    preds_test_df = preds_test_df[preds_test_df.Date != "2015-12-26"]
    preds_test_df = preds_test_df[preds_test_df.Date != "2015-12-27"]
    preds_test_df = preds_test_df[preds_test_df.Date != "2015-12-31"]
    result = preds_test_df['pred'].values
    
    return result



if __name__ == "__main__":    

    features_set_list = [['Adj Close'],
                         ['Adj Close','Sentiment Scores'],
                         ['Adj Close','Sentiment Scores','Change in %'],
                         ['Adj Close','Sentiment Scores','Change in %', 'year', 'month', 'date','day']]
    
    
    model_list= []
    model_history_list = []
    
    mse_train_list = []
    mse_val_list = []
    
    test_preds_list = []
    
    
    stock_nums = [0,1,2,3,4,5,6,7]
    for stock_num in stock_nums:
    
        # Read training Data
        tf.random.set_seed(13)
        all_dataset_dir = get_all_dataset_dir()
        all_dataset = read_all_dataset_dataframe(all_dataset_dir)
        df = all_dataset[stock_num].copy()
        
        
        df_obj = pd.DataFrame(columns = features_set_list[1])
        for f in features_set_list[1]:
            df_obj[f] = df[f]
            
        # Read text_tweet data
        tgt_tweet = get_tgt_tweet_dir()
        tgt_tweet_dfs = read_tgt_tweet_dataframe(tgt_tweet)
        tgt_tweet_df = tgt_tweet_dfs[stock_num]
        
        #splitData
        split_for_val = 72
        train, val = splitData(df_obj.copy(), split_for_val)
        
        
        #Normalization
        X_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaler.fit(df_obj)
        train_norm = X_scaler.transform(train)
        val_norm = X_scaler.transform(val)
        
        
        Y_scaler = MinMaxScaler(feature_range=(-1, 1))
        Y = np.array(df_obj.iloc[:]["Adj Close"])[:,np.newaxis]
        Y_scaler.fit(Y)
        
        
        #buildTimesetpData
        pastdays =10
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
        model = buildManyToOneModel(X_train_norm.shape,futherday)
        callback = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="auto")
        history  = model.fit(X_train_norm, Y_train_norm, epochs=500, batch_size=8, validation_data=(X_val_norm, Y_val_norm), callbacks=[callback])
        model_list.append(model)
        model_history_list.append(history)
        
        plot_model_loss(history,
                        "Modle Loss :(32)LSTM, pastdays=10 futherday = 1, stock_{}".format(stock_num))
        mse_train, mse_val = model_performance_visualization(
                        train_norm, 
                        val_norm,
                        train["Adj Close"].values[pastdays:][:,np.newaxis], 
                        val["Adj Close"].values[pastdays:][:,np.newaxis], 
                        pastdays, 
                        futherday, 
                        model,
                        "Model perfomance : (32)LSTM, pastdays=10 futherday = 1, stock_{}".format(stock_num))
    
        mse_train_list.append(mse_train)
        mse_val_list.append(mse_val)
    
        #pred test stock
        test_preds = test_inference(model,tgt_tweet_df,df_obj,X_scaler,Y_scaler,pastdays)
        test_preds_list.append(test_preds)
    
    




