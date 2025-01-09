# This is a sample Python script.
import datetime
import os
from urllib.request import urlopen, Request
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pypfopt import BlackLittermanModel
from pypfopt import black_litterman, risk_models
from scipy.optimize import minimize
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
import os

import pandas_datareader.data as we


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press ⌘F8 to toggle the breakpoint.


def genStocks():
    finviz_url = "https://finviz.com/quote.ashx?t="

    tickers = ['EUDA', 'ALAR', 'ANF', 'VITL', 'ADMA', 'APEI', 'AMSC', 'RNA', 'AEYE', 'CLS', 'ML', 'DAVE', 'EVER', 'FTAI',
               'WGS', 'DYN', 'GCT', 'SMR', 'MGNX', 'CURV', 'FIP']
    count =0
    for i in tickers:
        count+=1
    print(count)
    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker

        response = Request(url=url, headers={'user-agent': 'my-app'})
        res = urlopen(response)

        html = BeautifulSoup(res, 'html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    # for ticker in tickers2:
    # url = finviz_url + ticker
    # tickers2 = ['TPC', 'CAMT', 'AEYE''CVNA' 'GRPN', 'MOD', 'EVER', 'SHIP', 'NVDA', 'GATO'FTAI', 'GGAL', 'SLN','SMCI
    # ', 'IESC']
    # res = urlopen(response)

    # news_table = html.find(id='news-table')
    # news_tables[ticker] = news_table

    parsed_data = []
    vader = SentimentIntensityAnalyzer()
    for ticker, news_table in news_tables.items():
        totSum = 0
        num = 0
        for row in news_table.findAll('tr'):
            if (row.a):
                title = row.a.get_text()
                val = vader.polarity_scores(title)
                totSum += val['compound']
                num += 1
        avg = totSum / num
        parsed_data.append([ticker, avg])
        print(ticker)

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    df = pd.DataFrame(parsed_data, columns=['Ticker', 'Avg'])

    symbols = df.sort_values(by=['Avg'], ascending=False).head(5)['Ticker'].tolist()
    vals = df.sort_values(by=['Avg'], ascending=False).head(5)['Avg'].tolist()
    return symbols, vals
def monteCarloOptimization(tickers):
    numSym = len(tickers)
    list = []
    start = datetime.datetime(2021, 1, 1)
    end = datetime.datetime(2021, 12, 31)
    for ticker in tickers:
        val = yf.download(ticker,start='2022-1-1', end='2022-12-12')
        vals = val['Close']
        list.append(vals)
    print(list)
    stocks = pd.concat([list[0], list[1], list[2], list[3], list[4]])

    stocks.columns = [tickers[0], tickers[1], tickers[2], tickers[3], tickers[4]]
    returns = stocks / stocks.shift(1)

    log_return = np.log(1 + stocks.pct_change())

    numPort = 5000

    all_weights = np.zeros((numPort, numSym))

    ret_arr = np.zeros(numPort)
    vol_arr = np.zeros(numPort)
    sharpe_arr = np.zeros(numPort)
    for ind in range(numPort):
        random_weights = np.array(np.random.random(numSym))
        rebalance_weights = random_weights / np.sum(random_weights)
        all_weights[ind, :] = rebalance_weights

        for w in rebalance_weights:
            w *= log_return.mean()
            w *= 252
        sum =0
        for val in rebalance_weights:
            sum+=val
        ret_arr[ind] = sum

        exp_vol = np.sqrt(
            np.dot(
                rebalance_weights.T,
                np.dot(
                    np.cov(log_return) * 252,
                    rebalance_weights
                )
            )
        )
        vol_arr[ind] = exp_vol
        sharpe_arr[ind] = (ret_arr[ind] - .01 / vol_arr[ind])
    maxIndex = sharpe_arr.argmax()
    return all_weights[maxIndex,:]






def markowitzOptimization(tickers):
    list = []

    for ticker in tickers:
        val = yf.download(ticker, start='2022-1-1', end='2022-12-12')
        vals = val['Close']
        list.append(vals)
    print(list)
    stocks = pd.concat([list[0], list[1], list[2], list[3], list[4]])

    stocks.columns = [tickers[0], tickers[1], tickers[2], tickers[3], tickers[4]]
    returns = stocks / stocks.shift(1)
    log_return = np.log(returns)
    meanLogRet = log_return.mean()
    Sigma = np.cov(log_return)
    def negativeSR(w):
        w = np.array(w)
        R = np.sum((meanLogRet*w) * 252)
        V = np.sqrt(np.dot(w.T, np.dot(Sigma * 252, w)))
        SR = R/V
        return -1 * SR
    def checkSumToOne(w):
        return np.sum(w)-1
    w0 = [0.20, 0.20, 0.20, 0.20, 0.20]

    bounds = ((0,1),(0,1),(0,1),(0,1),(0,1))
    constraints = ({'type':'eq','fun':checkSumToOne})
    opt = minimize(negativeSR, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    print(opt)


def markowitzOptimization2(tickers):
    data = []

    # Download data
    for ticker in tickers:
        val = yf.download(ticker, start='2022-01-01', end='2022-12-12')
        data.append(val['Close'])

    # Concatenate data into a DataFrame
    stocks = pd.concat(data, axis=1)
    stocks.columns = tickers
    returns = stocks/stocks.shift(1)
    # Calculate log returns and drop NaNs
    log_return = np.log(returns)
    meanLogRet = log_return.mean()
    Sigma = log_return.cov()
    rM = .02
    N = len(Sigma)
    o = np.ones(N)
    sigmaInv = np.linalg.inv(Sigma)
    a = np.dot(meanLogRet.T, np.dot(sigmaInv, meanLogRet))
    b = np.dot(meanLogRet.T, np.dot(sigmaInv, o))
    c = np.dot(o.T, np.dot(sigmaInv, o))
    probs = softmax((1/(a*c - b**2)) * np.dot(sigmaInv, ((c * rM - b) * meanLogRet + (a - b * rM)*o)))
    return probs
    # Debugging prints

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def BlackLittermanOptimization(symbols, sentiments):
    portfolio = yf.download(symbols, start="2018-01-01", end="2023-2-20")['Adj Close']
    market_prices = yf.download('SPY', start="2018-01-01", end="2023-2-20")['Adj Close']
    mcaps = {}
    for t in symbols:
        stock = yf.Ticker(t)
        mcaps[t] = stock.info['marketCap']
    s = risk_models.CovarianceShrinkage(portfolio).ledoit_wolf()

    delta = black_litterman.market_implied_risk_aversion(market_prices)
    viewDict = {
        symbols[0]: sentiments[0],
        symbols[1]: sentiments[1],
        symbols[2]: sentiments[2],
        symbols[3]: sentiments[3],
        symbols[4]: sentiments[4]
    } # CONTAIN SENTIMENT VALUES FOR EACH STOCK

    variances = []
    for p1 in viewDict:
        sigma = viewDict[p1]
        variances.append(sigma**2)
    omega = np.diag(variances)
    bl = BlackLittermanModel(s, pi='market',market_caps = mcaps,risk_aversion=delta, absolute_views=viewDict, omega=omega  )
    return bl.bl_weights()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    monteClient = TradingClient(os.environ['MONTEKEY'], os.environ['MONTESECR'])
    blackLitClient = TradingClient(os.environ['BLACKKEY'],os.environ['BLACKSECR'] )
    markowClient = TradingClient(os.environ['MARKOWKEY'],os.environ['MARKOWSECR'])
    stocks, vals = genStocks()
    bidPrices = []
    amounts = []
    totCash = 100000
    import requests
    import time

    url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?{stocks[0], stocks[1], stocks[2], stocks[3], stocks[4]}&feed=sip"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)
    print(stocks)

    for stock in stocks:
        print("12344")
        account = monteClient.get_account()
        preC = account.buying_power
        pre = float(preC)
        marketOrder =MarketOrderRequest(symbol=stock, qty=1, side=OrderSide.BUY,
                                                                 time_in_force=TimeInForce.DAY)
        monteClient.submit_order(
            order_data=marketOrder
        )
        ctr =0;
        while True:
            print( f"Run {ctr}:")
            acc = monteClient.get_account()
            postC = acc.buying_power
            post = float(postC)
            if post - pre == 0:
                time.sleep(300)
            else:
                cost = pre-post
                bidPrices.append(cost)
                break;
            ctr+=1

    totCash = float(account.cash)
    split = monteCarloOptimization(stocks)
    print(len(bidPrices))
    for i in range(len(bidPrices)):
        val = totCash * split[i]
        print(val)
        amounts.append( val // bidPrices[i])
    print(sum(amounts))
    for i in range(len(amounts)):
        marketOrder =MarketOrderRequest(symbol=stocks[i], qty=amounts[i], side=OrderSide.BUY,
                                                                         time_in_force=TimeInForce.DAY)
        monteClient.submit_order(
                order_data=marketOrder
            )

    amounts.clear()
    split = markowitzOptimization2(stocks)
    account = markowClient.get_account()
    totCash = float(account.cash)
    for i in range(len(bidPrices)):
        val = totCash * split[i]
        amounts.append( val // bidPrices[i])

    for i in range(len(amounts)):
        marketOrder = MarketOrderRequest(symbol=stocks[i], qty=amounts[i], side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
        markowClient.submit_order(
                     order_data=marketOrder
                )



        amounts.clear()
        split = BlackLittermanOptimization(stocks,vals )
        account = blackLitClient.get_account()
        totCash = float(account.cash)
        ints = []
        print(split)
        for i in split:
            ints.append(split[i])

        print(ints)
        for i in range(len(bidPrices)):
            val = totCash * ints[i]
            amounts.append(val // bidPrices[i])
        for i in range(len(amounts)):
            marketOrder = MarketOrderRequest(symbol=stocks[i], qty=amounts[i], side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            blackLitClient.submit_order(
            order_data=marketOrder
        )










