# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 03:40:30 2019

@author: raymond
"""

import ast
import pandas as pd
import os
import datetime
import numpy as np

def get_all_preprocess_stock_dir():
    preprocess_stock_data_DIRs = []
    cwd = os.getcwd()
    stock_folder = os.path.join(cwd, 'stock_dataset_v3','preprocess_stock')
    preprocess_stock_data_DIRs = [os.path.join(stock_folder, f) for f in os.listdir(stock_folder)]
    return preprocess_stock_data_DIRs

def read_all_preprocess_stock_dataframe(stock_DIRs):
    all_preprocess_stock_data = []
    for stock_DIR in stock_DIRs:
        data = pd.read_csv(stock_DIR) 
        all_preprocess_stock_data.append(data)
    
    return all_preprocess_stock_data

def get_all_sentiment_analysis_dir():
    cwd = os.getcwd()
    preprocess_sentiment_analysis_dir = os.path.join(cwd, 'stock_dataset_v3','sentiment_analysis')
    preprocess_sentiment_analysis_dirs = [os.path.join(preprocess_sentiment_analysis_dir, f) for f in os.listdir(preprocess_sentiment_analysis_dir)]

    return preprocess_sentiment_analysis_dirs

def read_all_sentiment_analysis_dataframe(preprocess_sentiment_analysis_dirs):
    all_sentiment_analysis = []
    for preprocess_sentiment_analysis_dir in preprocess_sentiment_analysis_dirs:
        data = pd.read_csv(preprocess_sentiment_analysis_dir) 
        all_sentiment_analysis.append(data)
    
    return all_sentiment_analysis

def create_new_dfs():
    dfs=[]
    for i in range(8):
        start_date = datetime.date(2014, 1, 1)
        end_date   = datetime.date(2015, 12, 21)
        dates = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))]
        df = pd.DataFrame(dates, columns = ['Date']) 
        df['Date']=pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
        dfs.append(df)
    return dfs



def conbine_dfs(dfs,all_stock_dfs,sentiment_analysis_dfs):
    
    for df,all_stock_df,sentiment_analysis_df in zip(dfs,all_stock_dfs,sentiment_analysis_dfs):
        
        df["Adj Close"] = ""

#        df['polarity_scores_without_clearing'] = ''
        df['Sentiment Scores'] = ''
        
        for i in range(len(all_stock_df)):
            if len(df[df['Date'].str.contains(all_stock_df["Date"][i])]) == 1:
                index = int(df[df['Date']== all_stock_df["Date"][i]].index[0])
                df['Adj Close'][index] = all_stock_df["Adj Close"][i]
                
        for i in range(len(sentiment_analysis_df)):
            if len(df[df['Date'].str.contains(sentiment_analysis_df["Date"][i])]) == 1:
                index = int(df[df['Date']== sentiment_analysis_df["Date"][i]].index[0])
#                df['polarity_scores_without_clearing'][index] = sentiment_analysis_df['polarity_scores_without_clearing'][i]
                df['Sentiment Scores'][index] = sentiment_analysis_df['Sentiment Scores'][i]

    return dfs

def dealing_with_missing_data(dfs):
    for df in dfs:
#        df['polarity_scores_without_clearing'].replace('', 0, inplace=True)
        df['Sentiment Scores'].replace('', 0, inplace=True)
        
        df['Adj Close'].replace('', np.nan, inplace=True)
        df['Adj Close'] = df['Adj Close'].interpolate(method='linear',limit_direction ='both', axis=0)
        #df['Adj Close'] = pd.concat([df['Adj Close'].ffill(), df['Adj Close'].bfill()]).groupby(level=0).mean()
        #df.dropna(subset=['Adj Close'], inplace=True)
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
        df['Change in %'].fillna(value=0, inplace=True)
        
    return dfs


def augFeatures_date(dfs):
    for df in dfs:
      df["Date"] = pd.to_datetime(df["Date"])
      df["year"] = df["Date"].dt.year
      df["month"] = df["Date"].dt.month
      df["date"] = df["Date"].dt.day
      df["day"] = df["Date"].dt.dayofweek
    
    dfs = [df.drop('Date', 1) for df in dfs]
    return dfs


        
def reindex(dfs):
    temp=[]
    for df in dfs:
        df = df.reindex(columns=['Date','Adj Close','Change in %','Sentiment Scores'])
        temp.append(df)
    return temp



def shift(dfs):
    temp_dfs = []
    for df in dfs:
        
        df_sentiment = df['Sentiment Scores']
        df_other = df.drop(['Sentiment Scores'], axis=1)
        df_sentiment = df_sentiment.drop(df_sentiment.index[0])
        df_other = df_other.drop(df_other.index[-1])
        df_sentiment.reset_index(drop=True, inplace=True)
        df_other.reset_index(drop=True, inplace=True)
        df = pd.concat( [df_other, df_sentiment], axis=1) 
        temp_dfs.append(df)
    return temp_dfs


def save_dfs(dfs):
    for num, df in enumerate(dfs, start=1):
        cwd = os.getcwd()
        save_dir = os.path.join(cwd, 'stock_dataset_v3','conbine_stock_and_tweet')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        df.to_csv(os.path.join(save_dir,'conbine_stock_and_tweet_{}.csv'.format(num)), index = None, header=True)


if __name__ == "__main__":   
    
    stock_DIRs = get_all_preprocess_stock_dir()
    all_stock_dfs = read_all_preprocess_stock_dataframe(stock_DIRs)
    sentiment_analysis_DIRs = get_all_sentiment_analysis_dir()
    sentiment_analysis_dfs = read_all_sentiment_analysis_dataframe(sentiment_analysis_DIRs)
    
    dfs = create_new_dfs()
    new_dfs = conbine_dfs(dfs,all_stock_dfs,sentiment_analysis_dfs)
    new_dfs = dealing_with_missing_data(new_dfs)
    new_dfs = shift(new_dfs)
    
    new_dfs_aug = augFeatures_percentage_error(new_dfs)
    new_dfs_aug = reindex(new_dfs_aug)
    new_dfs_aug= augFeatures_date(new_dfs_aug)


    save_dfs(new_dfs_aug)
    

    
    