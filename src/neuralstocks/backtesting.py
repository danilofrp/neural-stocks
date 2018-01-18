from __future__ import division
import numpy as np
import pandas as pd
import random
from collections import defaultdict

class Backtest:
    dailyData = {}
    history = {}
    verbose = 0;
    shortedFunds = 0
    openPositions = []
    strategy = ''

    stopLoss = None
    stopGain = None
    maxExposure = None
    maxTotalExposure = None

    def __init__(self, assets, dataPath = '../../../data/stocks/[asset]/diario/[asset].CSV', funds = 100000,
                 brokerage = 6.0, transactionFees = 0.000325, ISStax = 0.05):
        self.assets = assets
        self.dataPath = dataPath
        self.funds = funds
        self.brokerage = brokerage
        self.transactionFees = transactionFees
        self.ISStax = ISStax

        self.loadData()

    def setRiskManagement(self, stopLoss = None, stopGain = None, maxExposure = None, maxTotalExposure = None):
        self.stopLoss = stopLoss
        self.stopGain = stopGain
        self.maxExposure = maxExposure
        self.maxTotalExposure = maxTotalExposure

    def loadData(self):
        for asset in self.assets:
            filePath = self.dataPath.replace('[asset]', asset)
            df = pd.read_csv(filePath, delimiter=';', decimal=',',
                     parse_dates=['Date'], dayfirst=True, index_col='Date').sort_index()
            self.dailyData[asset] = df

    def simulate(self, funds = None, strategy = 'buy-n-hold', start = None, end = None, longOnly = False, predicted = None, verbose = 0):
        self.strategy = strategy if strategy else self.strategy
        self.funds = funds if funds else self.funds
        self.verbose = verbose
        if predicted:
            for asset in assets:
                if not predicted[asset]:
                    print('Warning: No predictions for {}. This asset will be excluded from the simulation.'.format(asset))
                else:
                    self.dailyData[asset] = pd.concat([self.dailyData[asset], predicted[asset]], axis = 1)
        if strategy == 'buy-n-hold':
            self.buyNHold(start, end)
        else:
            pass
            #for i in range(len(predicted)):

    def buyNHold(self, start, end):
        self.history['buy-n-hold'] = pd.DataFrame(index=self.dailyData[self.assets[0]][start:end].index, columns=['portfolioValue'])
        maxValue = self.funds / len(self.assets)
        random.shuffle(self.assets)
        date = self.dailyData[self.assets[0]][start:end].index[0]
        print('Portfolio value at start: {:.2f} BRL'.format(self.funds))
        previousPortfolioValue = self.funds
        #buy all
        for asset in self.assets:
            self.buy(asset = asset, date = self.dailyData[asset][start:end].index[0], limitValue = maxValue)
        #evaluate portfolio value for each trading day
        for d in self.dailyData[self.assets[0]][start:end].index:
            date = d
            self.history['buy-n-hold'].at[date, 'portfolioValue'] = self.evaluatePortfolio(date)
            if self.verbose >= 2:
                print('Portfolio value at {} market close: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), self.history['buy-n-hold']['portfolioValue'][date]))
        #sell all
        for asset in self.assets:
            self.sell(asset = asset, date = self.dailyData[asset][start:end].index[-1])
        print('Portfolio value at end: {:.2f} BRL'.format(self.funds))
        self.calculateDrawdown()

    def buy(self, asset, date, volume = None, limitValue = None):
        price = self.dailyData[asset]['Open'][date]
        if not volume and not limitValue:
            print('{} - Error buying {} - Neither volume nor limitValue were specified'.format(day, asset))
        else:
            limitVolume = int(((limitValue/price)//100)*100)
            volume = volume if volume else limitVolume
            buyValue = volume * price
            fees = self.brokerage + (buyValue * self.transactionFees) + (self.brokerage * self.ISStax)
            self.openPositions.append(self.createPosition(asset, 'long', volume, price, date))
            self.funds -= (buyValue + fees)
            if self.verbose >= 2:
                print('{} Open - Bought {} {}. price: {} BRL, fees: {:.2f}, total: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), volume, asset, price, fees, buyValue + fees))

    def sell(self, asset, date):
        operation = self.openPositions.pop(findIndex(self.openPositions, asset, lambda x, y: x['asset'] == y))
        volume = operation['volume']
        price = self.dailyData[asset]['Close'][date]
        sellValue = volume * price
        fees = self.brokerage + (sellValue * self.transactionFees) + (self.brokerage * self.ISStax)
        self.funds += (sellValue - fees)
        if self.verbose >= 2:
            print('{} Close - Sold {} {}. price: {} BRL, fees: {:.2f}, total: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), volume, asset, price, fees, sellValue + fees))

    def createPosition(self, asset, opType, volume, price, date = None):
        return {'asset': asset, 'opType': opType, 'volume': volume, 'price': price, 'date': date}

    def evaluatePortfolio(self, date, moment = 'Close'):
        portfolio = self.funds + self.shortedFunds
        for l in filter(lambda x: x['opType'] == 'long', self.openPositions):
            portfolio += l['volume'] * self.dailyData[l['asset']][moment][date]
        for s in filter(lambda x: x['opType'] == 'short', self.openPositions):
            portfolio -= s['volume'] * self.dailyData[s['asset']][moment][date]
        return portfolio

    def calculateDrawdown(self):
        if self.history[self.strategy] is not None and self.history[self.strategy]['portfolioValue'] is not None:
            drawdown = (self.history[self.strategy]['portfolioValue'] - np.maximum.accumulate(self.history[self.strategy]['portfolioValue']))/np.maximum.accumulate(self.history[self.strategy]['portfolioValue'])
            self.history[self.strategy] = self.history[self.strategy].assign(drawdown = drawdown)


class Position:
    asset_index = defaultdict(object)

    def __init__(self, asset, opType, volume, price, date = None):
        self.asset = asset
        self.opType = opType
        self.volume = volume
        self.price = price
        self.date = date
        Position.asset_index[asset] = self

    def __getitem__(self, key):
        if key == 'asset':
            return self.asset
        elif key == 'opType':
            return self.opType
        elif key == 'volume':
            return self.volume
        elif key == 'price':
            return self.price
        elif key == 'date':
            return self.date
        else:
            return None

    @classmethod
    def findByAsset(cls, asset):
        return Position.asset_index[asset]

    @classmethod
    def getByAsset(cls, asset):
        return Position.asset_index.pop(asset, None)

def findIndex(array, obj, func):
    for i in range(len(array)):
        if func(array[i], obj):
            return i
    return -1
