#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:17:24 2019

@author: tomoki
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

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



def augFeatures_percentage_error(now,previous):
    change = 100*(now - previous) / previous
    return change

def augFeatures_date(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["date"] = df["Date"].dt.day
    df["day"] = df["Date"].dt.dayofweek
    return df





def test_inference(model,tgt_tweet_df,df_obj,X_scaler,Y_scaler):
    
    tgt_date = tgt_tweet_df["Date"].values
    
    
    if "Sentiment Scores" not in df_obj.keys():
        
        X_start = df_obj['Adj Close'][df_obj.index[-1]].reshape(-1,1)
        X_start = X_scaler.transform(X_start)
        
        pred = X_start
        preds = []
        for i in range(len(tgt_date)):
            pred = model.predict(pred.reshape(-1,1))
            preds.append(pred.squeeze())
    else:
        X_start = df_obj['Adj Close'][df.index[-1]].reshape(-1,1)
        X_tweet = df_obj["Sentiment Scores"][df_obj.index[-1]].reshape(-1,1)
        X_start = np.hstack((X_start,X_tweet))
        X_start = X_scaler.transform(X_start)
        
        tgt_sentiment = tgt_tweet_df["Sentiment Scores"].values
        tgt_sentiment = tgt_sentiment.reshape(-1,1)
        pad = np.zeros(tgt_sentiment.shape[0]).reshape(-1,1)
        tgt_sentiment = np.concatenate((pad,tgt_sentiment), axis=1)
        tgt_sentiment= X_scaler.transform(tgt_sentiment)[:,1]
        
        
        pred = X_start
        preds = []
        for sentiment in tgt_sentiment:
            
            pred = model.predict(pred)
            preds.append(pred.squeeze())
            pred = np.hstack((pred.reshape(-1,1),sentiment.reshape(-1,1)))
            
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
    
    stock_nums = [0,1,2,3,4,5,6,7]
    
    features_set_list = [['Adj Close'],
                     ['Adj Close','Sentiment Scores'],
                     ['Adj Close','Sentiment Scores','Change in %'],
                     ['Adj Close','Sentiment Scores','Change in %', 'year', 'month', 'date','day']]
    
    features_set = [0,0,0,0,0,0,0,0]
    
    
    model_configs =[{'C': 1000, 'kernel': 'linear'},
                   {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'},
                   {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'},
                   {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'},
                   {'C': 1000, 'kernel': 'linear'},
                   {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'},
                   {'C': 1000, 'kernel': 'linear'},
                   {'C': 1, 'kernel': 'linear'}]
    
    val_mse_list = []
    train_mse_list = []
    
    test_preds_list = []

    for stock_num,features,model_config in zip(stock_nums, features_set, model_configs):
        
        # Read text_tweet data
        tgt_tweet = get_tgt_tweet_dir()
        tgt_tweet_dfs = read_tgt_tweet_dataframe(tgt_tweet)
        tgt_tweet_df = tgt_tweet_dfs[stock_num]
        
        all_dataset_dir = get_all_dataset_dir()
        all_dataset = read_all_dataset_dataframe(all_dataset_dir)
        df = all_dataset[stock_num].copy()
        
        # select features
        df_obj = pd.DataFrame(columns = features_set_list[features])
        for f in features_set_list[features]:
            df_obj[f] = df[f]
        
        # データの読み込み
        X = df_obj
        X = X.drop(X.index[-1])
        if len(X.shape)>1:
            X = X.values.reshape(-1,X.shape[1])
        else:
            X = X.values.reshape(-1,1)
        
        y = df_obj['Adj Close']
        y = y.drop(y.index[0])
        y = y.values.reshape(-1,1)
    
        # 訓練データ、テストデータに分割
        X, Xtest, y, ytest = train_test_split(X, y, test_size=0.05, random_state=114514)
        
        X_scaler =  MinMaxScaler(feature_range=(-1, 1))
        X_scaler.fit(X)
        Y_scaler =  MinMaxScaler(feature_range=(-1, 1))
        Y_scaler.fit(y)
        X_norm = X_scaler.transform(X)
        y_norm = Y_scaler.transform(y)
        
        model = SVR(**model_config)
        model.fit(X_norm,y_norm.reshape(-1))
        
        
        y_pred_train = model.predict(X_norm)  
        y_pred_train = Y_scaler.inverse_transform(y_pred_train.reshape(-1,1))
        train_mse = mean_squared_error(y, y_pred_train)
        train_mse_list.append(train_mse)
        
        
        Xtest_norm = X_scaler.transform(Xtest)
        y_pred_test = model.predict(Xtest_norm)  
        y_pred_test = Y_scaler.inverse_transform(y_pred_test.reshape(-1,1))
        val_mse = mean_squared_error(ytest, y_pred_test)
        val_mse_list.append(val_mse)
        
        test_preds = test_inference(model,tgt_tweet_df,df_obj,X_scaler,Y_scaler)
        test_preds_list.append(test_preds)
