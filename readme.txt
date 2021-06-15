Before model training, do the data preparation

1.preprocess_tweet_data.py ------> read all tweet data and save as csv dataframe format in stock_dataset_v3/preprocess_tweet
2.sentiment_analysis.py ---------> read all preprocess tweet data and tranform to sentiment sorces and save as stock_dataset_v3/sentiment_analysis
3.preprocess_stock_data.py-------> read all stock data save as csv dataframe format in stock_dataset_v3/preprocess_stock
4.conbine_stock_and_tweet.py-----> read all stock data and sentiment sorces data and conbine them to single datafram and save as stock_dataset_v3/conbine_stock_and_tweet 

Next,like training stock and tweet data, do the data preparation of test data

5.preprocess_tweet_data_for_test
6.sentiment_analysis_for_test


Finally, model training process

7.model_selection_SVR.py------> serach hyper parameter of SVR
8.train_SVR.py  --------------> SVR training and inference test data

9.model_selection_LSTM.py-----> serach hyper parameter of LSTM
10.train_LSTM.py--------------> LSTM training and inference test data

11.dataFile.pkl  -------------> prediction of test data 

Total 10 python script in this project,and the file of prediction of test data 