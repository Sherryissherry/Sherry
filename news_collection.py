import finnhub
import sys
import os
import time

from pandas import json_normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.common import *

finnhub_client = finnhub.Client(api_key="cfja5jhr01que34nsrugcfja5jhr01que34nsrv0")

sp500_tickers = get_sp500_tickers()
for ticker in sp500_tickers:
    if os.path.exists(ticker + ".csv"):
        continue
    time.sleep(1)
    news = finnhub_client.company_news(ticker, _from="2022-03-02", to="2023-03-01")
    df = json_normalize(news)
    df.reset_index(drop=True).to_csv("data/backtest/" + ticker + ".csv")

