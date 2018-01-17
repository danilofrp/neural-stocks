from __future__ import division
import numpy as np
import pandas as pd
import random
from collections import defaultdict

class Backtest:
    dailyData = {}
    history = {}
    verbose = 0;

    stopLoss = None
    stopGain = None
    maxExposure = None
    maxTotalExposure = None
    openOperations = []

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

    def createOperation(self, asset, opType, volume, price):
        return {'asset': asset, 'opType': opType, 'volume': volume, 'price': price}

    def simulate(self, funds = None, strategy='buy-n-hold', start = None, end = None, longOnly = False, predicted = None, verbose = 0):
        self.funds = funds if funds else self.funds
        self.verbose = verbose
        if predicted:
            for asset in assets:
                self.dailyData[asset] = pd.concat([self.dailyData[asset], predicted[asset]], axis = 1)
        if strategy == 'buy-n-hold':
            self.buyNHold(start, end)
        else:
            pass
            #for i in range(len(predicted)):

    def buyNHold(self, start, end):
        maxValue = self.funds / len(self.assets)
        random.shuffle(self.assets)
        print('Portfolio value: {}'.format(self.funds))
        for asset in self.assets:
            self.buy(asset = asset, day = self.dailyData[asset][start:end].index[0], limitValue = maxValue)
        for asset in self.assets:
            self.sell(asset = asset, day = self.dailyData[asset][start:end].index[-1])
        print('Portfolio value: {}'.format(self.funds))


    def buy(self, asset, day, volume = None, limitValue = None):
        price = self.dailyData[asset]['Open'][day]
        if not volume and not limitValue:
            print('{} - Error buying {} - Neither volume nor limitValue were specified'.format(day, asset))
        else:
            limitVolume = int(((limitValue/price)//100)*100)
            volume = volume if volume else limitVolume
            buyValue = volume * price
            fees = self.brokerage + (buyValue * self.transactionFees) + (self.brokerage * self.ISStax)
            self.openOperations.append(self.createOperation(asset, 'long', volume, price))
            self.funds -= (buyValue + fees)
            if self.verbose >= 2:
                print('{} Open - Bought {} {}. price: {}, fees: {:.2f}, total: {:.2f}'.format(day.strftime('%Y-%m-%d'), volume, asset, price, fees, buyValue + fees))

    def sell(self, asset, day):
        operation = self.openOperations.pop(findIndex(self.openOperations, asset, lambda x, y: x['asset'] == y))
        volume = operation['volume']
        price = self.dailyData[asset]['Close'][day]
        sellValue = volume * price
        fees = self.brokerage + (sellValue * self.transactionFees) + (self.brokerage * self.ISStax)
        self.funds += (sellValue - fees)
        if self.verbose >= 2:
            print('{} Close - Sold {} {}. price: {}, fees: {:.2f}, total: {:.2f}'.format(day.strftime('%Y-%m-%d'), volume, asset, price, fees, sellValue + fees))

class Operation:
    op_index = defaultdict(list)

    def __init__(self, asset, opType, volume, price):
        self.asset = asset
        self.opType = opType
        self.volume = volume
        self.price = price
        Operation.op_index[asset].append(self)

    @classmethod
    def findByAsset(cls, asset):
        return Operation.op_index[asset]

def findIndex(array, item, func):
    for i in range(len(array)):
        if func(array[i], item):
            return i
    return -1
