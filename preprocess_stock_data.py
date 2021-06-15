# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:50:53 2019

@author: raymond
"""

import ast
import pandas as pd
import os
import numpy as np

def get_all_stock_dir():
    stock_data_DIRs = []
    cwd = os.getcwd()
    stock_folder = os.path.join(cwd, 'stock_dataset_v3','raw_price_train')
    stock_data_DIRs = [os.path.join(stock_folder, f) for f in os.listdir(stock_folder)]
    return stock_data_DIRs

def read_all_stock_dataframe(stock_DIRs):
    all_stock_data = []
    for stock_DIR in stock_DIRs:
        data = pd.read_csv(stock_DIR) 
        data.dropna(subset=['Date'], inplace=True)
        all_stock_data.append(data)

    return all_stock_data


def augFeatures_date(dfs):
    for df in dfs:
      df["Date"] = pd.to_datetime(df["Date"])
      df["year"] = df["Date"].dt.year
      df["month"] = df["Date"].dt.month
      df["date"] = df["Date"].dt.day
      df["day"] = df["Date"].dt.dayofweek
    return dfs

def augFeatures_percentage_error(dfs):
    for df in dfs:
        stock_change = [np.nan]
        for i in range(len(df)-1):
            now = df["Adj Close"][i+1]
            previous = df["Adj Close"][i]
            change = 100*(now - previous) / previous
            stock_change.append(change)
        df['Change in %'] = pd.Series(stock_change).values
    return dfs

def change_date_format(dfs):
    for df in dfs:
        df['Date']=pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
    return dfs

def save_dfs(dfs):
    for num, df in enumerate(dfs, start=1):
        cwd = os.getcwd()
        save_dir = os.path.join(cwd, 'stock_dataset_v3','preprocess_stock')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        df.to_csv(os.path.join(save_dir,'preprocess_stock_{}.csv'.format(num)), index = None, header=True)


if __name__ == "__main__":    

    stock_DIRs = get_all_stock_dir()
    all_stock_dfs = read_all_stock_dataframe(stock_DIRs)
    
    all_stock_dfs = [all_stock_df.drop('Open', 1) for all_stock_df in all_stock_dfs]
    all_stock_dfs = [all_stock_df.drop('High', 1) for all_stock_df in all_stock_dfs]
    all_stock_dfs = [all_stock_df.drop('Low', 1) for all_stock_df in all_stock_dfs]
    all_stock_dfs = [all_stock_df.drop('Volume', 1) for all_stock_df in all_stock_dfs]
    all_stock_dfs = [all_stock_df.drop('Close', 1) for all_stock_df in all_stock_dfs]
    
    all_stock_dfs = augFeatures_percentage_error(all_stock_dfs)
    all_stock_dfs = augFeatures_date(all_stock_dfs)
    all_stock_dfs = change_date_format(all_stock_dfs)
    save_dfs(all_stock_dfs)
