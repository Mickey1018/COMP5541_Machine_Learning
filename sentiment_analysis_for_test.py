# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:35:09 2019

@author: raymond
"""

import ast
import pandas as pd
import os
import re, string
import numpy as np


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_all_preprocess_tweet_dir():
    cwd = os.getcwd()
    preprocess_tweet_dir = os.path.join(cwd, 'tweet_test','preprocess_tweet_test')
    preprocess_tweet_dirs = [os.path.join(preprocess_tweet_dir, f) for f in os.listdir(preprocess_tweet_dir)]

    return preprocess_tweet_dirs

def read_all_preprocess_tweet_dataframe(preprocess_tweet_DIRs):
    all_preprocess_tweet = []
    for preprocess_tweet_DIR in preprocess_tweet_DIRs:
        data = pd.read_csv(preprocess_tweet_DIR) 
        all_preprocess_tweet.append(data)
    
    return all_preprocess_tweet

def remove_noise(tweet_tokens, stop_words = stopwords.words('english')):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        
        token = re.sub("rt","", token)
        token = re.sub("URL","", token)
        token = re.sub("AT_USER","", token)
        
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_polarity_scores_tweet_tokens_nltk(tweet_dfs):
    sid = SentimentIntensityAnalyzer()

    all_polarity_scores_all_day = []
    for tweet_df in tweet_dfs:
        
        polarity_scores_all_day = []
        
        for tweet_str in tweet_df['Tweet']:
            tweet_tokens = ast.literal_eval(tweet_str)
            polarity_scores_one_day = []
            
            for tweet_token in tweet_tokens:
                polarity_scores_one_day.append(sid.polarity_scores(" ".join(tweet_token))['compound'])
            polarity_scores_all_day.append(np.mean(polarity_scores_one_day))
            
        all_polarity_scores_all_day.append(polarity_scores_all_day)
        
    return all_polarity_scores_all_day


def get_polarity_scores_tweet_cleaned_tokens_nltk(tweet_dfs):
    sid = SentimentIntensityAnalyzer()

    all_polarity_scores_all_day = []
    for tweet_df in tweet_dfs:
        polarity_scores_all_day = []
        
        for tweet_str in tweet_df['Tweet']:
            tweet_tokens = ast.literal_eval(tweet_str)
            
            tweet_clearn_tokens = []
            for tweet_token in tweet_tokens:
                tweet_clearn_tokens.append(remove_noise(tweet_token))
            
        
            polarity_scores_one_day = []
            for tweet_clearn_token in tweet_clearn_tokens:
                polarity_scores_one_day.append(sid.polarity_scores(" ".join(tweet_clearn_token))['compound'])
        
            polarity_scores_all_day.append(np.mean(polarity_scores_one_day))
            
        all_polarity_scores_all_day.append(polarity_scores_all_day)
        
    return all_polarity_scores_all_day


def add_polarity_scores_to_dataframe(tweet_dfs, column_name, polarity_scores):
    
    for tweet_df, polarity_score in zip(tweet_dfs, polarity_scores):
        tweet_df[column_name] = pd.Series(polarity_score).values
        
    return tweet_dfs


def save_sentiment_analysis_dataframes(sentiment_analysis_dataframes):
    for num, tweet_dataframe in enumerate(sentiment_analysis_dataframes, start=1):
        cwd = os.getcwd()
        save_dir = os.path.join(cwd, 'tweet_test','sentiment_analysis_test')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tweet_dataframe.to_csv(os.path.join(save_dir,'sentiment_analysis_{}.csv'.format(num)), index = None, header=True)
        
        
        
if __name__ == "__main__":    

    preprocess_tweet_DIRs = get_all_preprocess_tweet_dir()
    tweet_dfs = read_all_preprocess_tweet_dataframe(preprocess_tweet_DIRs)
    
    all_polarity_scores_tweet_tokens_nltk = get_polarity_scores_tweet_tokens_nltk(tweet_dfs)
    all_polarity_scores_tweet_cleaned_tokens_nltk = get_polarity_scores_tweet_cleaned_tokens_nltk(tweet_dfs)
    
    sentiment_analysis_dfs = add_polarity_scores_to_dataframe(tweet_dfs, 'Sentiment Scores', all_polarity_scores_tweet_cleaned_tokens_nltk)
    sentiment_analysis_dfs = [sentiment_analysis_df.drop('Tweet', 1) for sentiment_analysis_df in sentiment_analysis_dfs]
    save_sentiment_analysis_dataframes(sentiment_analysis_dfs)
    




