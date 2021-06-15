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
from sklearn.model_selection import GridSearchCV
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


def save_to_csv(all_train_mse_list,all_val_mse_list,all_best_params_list):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd,'SVR_model_selection')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for num, (train_mse_list,val_mse_list,best_params_list) in enumerate(zip(all_train_mse_list,all_val_mse_list,all_best_params_list), start=1):
        
        data = {'training loss':train_mse_list, 'validation':val_mse_list, 'best params':best_params_list} 
        df = pd.DataFrame(data, index =['feature set 1', 'feature set 2', 'feature set 3', 'feature set 4']) 
    
        df.to_csv(os.path.join(save_dir,'SVR_model_selection_{}.csv'.format(num)), index = None, header=True)




if __name__ == "__main__":   

    stock_nums = [0,1,2,3,4,5,6,7]
    
    features_set_list = [['Adj Close'],
                     ['Adj Close','Sentiment Scores'],
                     ['Adj Close','Sentiment Scores','Change in %'],
                     ['Adj Close','Sentiment Scores','Change in %', 'year', 'month', 'date','day']]
    
    all_train_mse_list = []
    all_val_mse_list = []
    all_best_params_list = []
    
    for stock_num in stock_nums:
        
        
        all_dataset_dir = get_all_dataset_dir()
        all_dataset = read_all_dataset_dataframe(all_dataset_dir)
        df = all_dataset[stock_num].copy()
        
        
        val_mse_list = []
        train_mse_list = []
        best_params_list = []
        for features_set in features_set_list:
            
            # select features
            df_obj = pd.DataFrame(columns = features_set)
            for f in features_set:
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
            
            param_grid = [
                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3, 4, 5, 6],'gamma':[1, 0.1, 0.01, 0.001, 0.0001]},
                {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001,0.0001]}
                    ]
        
            model = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3) 
            model.fit(X_norm,y_norm.reshape(-1))
            best_params_list.append(model.best_params_)
             
            y_pred_train = model.predict(X_norm)  
            y_pred_train = Y_scaler.inverse_transform(y_pred_train.reshape(-1,1))
            train_mse = mean_squared_error(y, y_pred_train)
            train_mse_list.append(train_mse)
            
            
            Xtest_norm = X_scaler.transform(Xtest)
            y_pred_test = model.predict(Xtest_norm)  
            y_pred_test = Y_scaler.inverse_transform(y_pred_test.reshape(-1,1))
            val_mse = mean_squared_error(ytest, y_pred_test)
            val_mse_list.append(val_mse)
        
    
        all_val_mse_list.append(val_mse_list)
        all_train_mse_list.append(train_mse_list)
        all_best_params_list.append(best_params_list)
    
    save_to_csv(all_train_mse_list,all_val_mse_list,all_best_params_list)