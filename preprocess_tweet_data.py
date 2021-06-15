# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:50:53 2019

@author: raymond
"""

import ast
import pandas as pd
import os



def get_all_tweet_dir():
    tweet_data_DIRs = []
    cwd = os.getcwd()
    tweet_DIR = os.path.join(cwd, 'stock_dataset_v3','tweet_train')
    tweet_folders = [os.path.join(tweet_DIR, f) for f in os.listdir(tweet_DIR)]
    
    for tweet_folder in tweet_folders:
        tweet_data_DIRs.append([os.path.join(tweet_folder, f) for f in os.listdir(tweet_folder)])
    return tweet_data_DIRs

def get_all_tweet_data(tweet_data_DIRs):
    all_tweet = []
    for tweet_data_DIR in tweet_data_DIRs:
        temp = []
        for tweet in tweet_data_DIR:
            s = open(tweet, 'r').read()
            tweet_raw = s.splitlines()
            tweet_dic = [ast.literal_eval('{%s}' % item[1:-1]) for item in tweet_raw]
            tweet_df = pd.DataFrame.from_dict(tweet_dic)
            tweet_text = list((tweet_df['text']))
            temp.append(tweet_text)
        all_tweet.append(temp)
    return all_tweet

def get_all_date_of_tweet(tweet_data_DIRs):
    all_date_of_tweet = []
    for tweet_data_DIR in tweet_data_DIRs:
        date_of_tweet = []
        for tweet_path in tweet_data_DIR:
            tweet_filename = os.path.basename(tweet_path)
            date_of_tweet.append(tweet_filename)
        all_date_of_tweet.append(date_of_tweet)
    return all_date_of_tweet

def to_dataframe(all_date_of_tweet,all_tweet_data):
    df_all_tweet=[]
    for date_of_tweet,tweet_data in zip(all_date_of_tweet,all_tweet_data):
        df_tweet = pd.DataFrame({'Date':date_of_tweet, 'Tweet':tweet_data} ) 
        df_all_tweet.append(df_tweet)
    return df_all_tweet
        
def save_tweet_dataframes(tweet_dataframes):
    for num, tweet_dataframe in enumerate(tweet_dataframes, start=1):
        cwd = os.getcwd()
        save_dir = os.path.join(cwd, 'stock_dataset_v3','preprocess_tweet')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tweet_dataframe.to_csv(os.path.join(save_dir,'preprocess_tweet_{}.csv'.format(num)), index = None, header=True)
        
        
        
if __name__ == "__main__":        
        
    tweet_data_DIRs = get_all_tweet_dir()
    all_date_of_tweet = get_all_date_of_tweet(tweet_data_DIRs)
    all_tweet_data = get_all_tweet_data(tweet_data_DIRs)
    all_tweet_dataframe = to_dataframe(all_date_of_tweet,all_tweet_data)
    save_tweet_dataframes(all_tweet_dataframe)
