import finnhub
import sys
import os
import time

from pandas import json_normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.common import *

sp500_tickers = get_sp500_tickers()
news = pd.read_csv("data/train/analyst_ratings_processed.csv", index_col = 0)
news['date'] = news['date'].str[:10]
news = news[news['stock'].isin(sp500_tickers)]
news = news[~news['stock'].isin(INVALID_SP500)]
news = news.sort_values(['date', 'stock'], ignore_index=True)


df_list = []

data_collection_states = pd.read_csv(DATA_COLLECTION_STATES_FILE, index_col = 0)
for p in data_collection_states["Processed data file"].tolist():
      print(p)
      df = pd.read_csv(p, index_col=0)
      df.replace([np.inf, -np.inf], 0, inplace=True)
      df_list.append(df)
all_data = pd.concat(df_list, ignore_index=True)
all_data = all_data[~all_data['ticker'].isin(INVALID_SP500)]
all_data = all_data[all_data['date'] >= "2010-01-01"]
all_data = all_data[all_data['date'] <= "2021-01-01"]
all_data = all_data.drop_duplicates(ignore_index = True)
all_data = all_data.sort_values(['date', 'ticker'], ignore_index=True)
all_data = all_data[["date", "ticker", "open", "close"]]

news_and_prices = pd.merge(news, all_data,  how='inner', left_on=['date','stock'], right_on = ['date','ticker'])
news_and_prices = news_and_prices[['date', 'ticker', 'title', 'open', 'close']]

def get_sentiment(row):
    change_rate = 0
    if row['open'] != 0:
       change_rate = (row['close'] - row['open']) / row['open']
       
    sentiment = 'neutral'
    if change_rate > 0.01:
        sentiment = 'positive'
    if change_rate < -0.01:
        sentiment = 'negative'
    
    return sentiment

news_and_prices['label'] = news_and_prices.apply(get_sentiment, axis=1)
news_and_prices = news_and_prices.rename(columns={"title": "sentence"})
news_and_prices = news_and_prices[['sentence', 'label']]
train = news_and_prices.sample(frac = 0.8)
validate = news_and_prices.drop(train.index)
train.reset_index(drop=True).to_csv("train.csv")
validate.reset_index(drop=True).to_csv("validation.csv")