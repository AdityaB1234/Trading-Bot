import alpaca.trading.requests
from alpaca.trading.client import TradingClient
from main import markowitzOptimization2
from main import BlackLittermanOptimization
from main import monteCarloOptimization
from main import genStocks
from alpaca.trading.enums import OrderSide, TimeInForce
monteClient = TradingClient("PK0XC7P60PSKG3NSWR3O","7Po2zGwtwlafXSJtDavTtDzV7qlGnQ68p6scSlnp")
markowClient= TradingClient()
blackLitClient = TradingClient()
from alpaca.trading.client import TradingClient
stocks = genStocks()
bidPrices = []
amounts = []
totCash = 100,000
import requests
import time
url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?{stocks[0], stocks[1], stocks[2], stocks[3], stocks[4]}&feed=sip"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)
for stock in stocks:

        account = monteClient.get_account()
        preC = account.buying_power
        marketOrder = alpaca.trading.requests.MarketOrderRequest(symbol=stock, qty=1, side=OrderSide.Buy, time_in_force=TimeInForce.DAY)
        monteClient.submit_order(
           order_data=marketOrder
       )
        while True:
            acc = monteClient.get_account()
            postC = acc.buying_power
            if postC - preC == 0:
                time.sleep(300)
            else:
                bidPrices.append(preC - postC)
                break;




split = monteCarloOptimization(stocks)
for i in range(len(stocks)):
    val = totCash * split[i]
    amounts[i] = val // bidPrices[i]

for i in range(len(amounts)):
    marketOrder = alpaca.trading.requests.MarketOrderRequest(symbol=stocks[i], qty=amounts[i], side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
    monteClient.submit_order(
        order_data=marketOrder
    )


amounts.clear()
# #
# split = markowitzOptimization2(stocks)
# for i in range(len(stocks)):
#     val = totCash * split[i]
#     amounts[i] = val // bidPrices[i]
#
# for i in range(len(amounts)):
#     marketOrder = alpaca.trading.requests.MarketOrderRequest(symbol=stocks[i], qty=amounts[i], side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
#     markowClient.submit_order(
#         order_data=marketOrder
#     )
#
# amounts.clear()
#
#
# split = BlackLittermanOptimization(stocks)
# for i in range(len(stocks)):
#     val = totCash * split[i]
#     amounts[i] = val // bidPrices[i]
#
# for i in range(len(amounts)):
#     marketOrder = alpaca.trading.requests.MarketOrderRequest(symbol=stocks[i], qty=amounts[i], side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
#     blackLitClient.submit_order(
#         order_data=marketOrder
#     )
#
