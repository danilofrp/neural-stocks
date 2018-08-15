# -*- coding: utf-8 -*-
from __future__ import division, print_function
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from datetime import timedelta
from matplotlib.ticker import EngFormatter

class Backtest:
    oneDay = timedelta(days = 1)

    def __init__(self, assets, dataPath = '../../../data/stocks/[asset]/diario/[asset].CSV', initialFunds = 100000,
                 brokerage = 6.0, transactionFees = 0.000325, ISStax = 0.05):
        self.assets = assets
        self.assetsToUse = assets
        self.dataPath = dataPath
        self.funds = initialFunds
        self.initialFunds = initialFunds
        self.brokerage = brokerage
        self.transactionFees = transactionFees
        self.ISStax = ISStax

        self.history = {}
        self.dailyData = {}
        self.verbose = 0;
        self.shortedFunds = 0
        self.pendingOperations = []
        self.openPositions = []
        self.strategy = ''
        self.predictedValues = {}
        self.orders = {}
        self.confidenceLimit = 0.0

        self.useRiskManagement = False
        self.stopLoss = None
        self.stopGain = None
        self.maxExposure = None
        self.maxTotalExposure = None

        self.loadData()

    def setRiskManagement(self, stopLoss = None, stopGain = None, maxExposure = None, maxTotalExposure = None, useRiskManagement = True):
        self.useRiskManagement = useRiskManagement
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

    def simulate(self, initialFunds, strategy = 'buy-n-hold', start = None, end = None, longOnly = False,
                 predicted = None, orders = None, confidenceLimit = 0.0,
                 simulationName = None, assetsToUse = None, verbose = 0):
        self.assetsToUse = self.assets if not assetsToUse else assetsToUse
        simulationName = simulationName if simulationName else strategy
        print('Starting {} simulation with {}'.format(simulationName, self.assetsToUse))
        self.strategy = strategy
        self.funds = initialFunds
        self.initialFunds = initialFunds
        self.confidenceLimit = confidenceLimit
        self.verbose = verbose
        if strategy == 'buy-n-hold':
            self.buyNHold(start, end)
        elif strategy == 'repeatLast':
            pass
        elif strategy == 'predicted':
            if predicted:
                assetsToRemove = []
                for asset in self.assetsToUse:
                    if not asset in predicted:
                        assetsToRemove.append(asset)
                        #del self.assetsToUse[asset]
                        print('Warning: No predictions for {}. This asset will be excluded from the simulation.'.format(asset))
                    else:
                        self.predictedValues[asset] = predicted[asset]
                self.assetsToUse = list(set(self.assetsToUse).symmetric_difference(set(assetsToRemove)))
            if self.predictedValues:
                self.simulatePredicted(start, end, simulationName)
            else:
                print('No predicted values given for the selected assets')
            self.predictedValues = {}
        elif strategy == 'orders':
            if orders:
                assetsToRemove = []
                for asset in self.assetsToUse:
                    if not asset in orders:
                        assetsToRemove.append(asset)
                        #del self.assetsToUse[asset]
                        print('Warning: No orders for {}. This asset will be excluded from the simulation.'.format(asset))
                    else:
                        self.orders[asset] = orders[asset]
                self.assetsToUse = list(set(self.assetsToUse).symmetric_difference(set(assetsToRemove)))
            if self.orders:
                self.simulateOrders(start, end, simulationName)
            else:
                print('No orders cue given for the selected assets')
            self.predictedValues = {}
        else:
            print('Selected strategy not recognized')

    def buyNHold(self, start, end):
        self.history['buy-n-hold'] = pd.DataFrame(index=self.dailyData[self.assetsToUse[0]][start:end].index, columns=['portfolioValue'])
        maxValue = self.funds / len(self.assetsToUse)
        random.shuffle(self.assetsToUse)
        date = self.dailyData[self.assetsToUse[0]][start:end].index[0]
        print('Portfolio value at start: {:.2f} BRL'.format(self.funds))
        previousPortfolioValue = self.funds
        #buy all
        for asset in self.assetsToUse:
            self.buy(asset = asset, date = self.dailyData[asset][start:end].index[0], limitValue = maxValue)
        self.history['buy-n-hold'].at[date, 'portfolioValue'] = self.funds
        #evaluate portfolio value for each trading day
        for d in self.dailyData[self.assetsToUse[0]][start:end].index:
            date = d
            self.history['buy-n-hold'].at[date, 'portfolioValue'] = self.evaluatePortfolio(date)
            if self.verbose >= 1:
                print('Portfolio value at {} market close: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), self.history['buy-n-hold']['portfolioValue'][date]))
        #sell all
        self.liquidateAll(date)
        print('Portfolio value at end: {:.2f} BRL'.format(self.funds))
        self.calculateDrawdown('buy-n-hold')

    def simulatePredicted(self, start, end, simulationName = 'predicted'):
        self.history[simulationName] = pd.DataFrame(index=self.dailyData[self.assetsToUse[0]][start:end].index, columns=['portfolioValue'])
        maxValue = self.funds / len(self.assetsToUse)
        date = self.dailyData[self.assetsToUse[0]][start:end].index[0]
        print('Portfolio value at start: {:.2f} BRL'.format(self.funds))
        self.history[simulationName].at[date, 'portfolioValue'] = self.funds
        # simulate for every day in the simulation period
        for d in self.dailyData[self.assetsToUse[0]][start:end].index:
            date = d
            #self.history[simulationName].at[date, 'portfolioValue'] = self.evaluatePortfolio(date)
            self.decideOperations(date)
            self.executeOperations(maxValue)
            self.liquidateAll()
            maxValue = self.funds / len(self.assetsToUse)
            self.history[simulationName].at[date, 'portfolioValue'] = self.evaluatePortfolio(date)
            if self.verbose >= 1:
                print('Portfolio value at {} market close: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), self.history[simulationName]['portfolioValue'][date]))
        print('Portfolio value at end: {:.2f} BRL'.format(self.funds))
        self.calculateDrawdown(simulationName)

    def decideOperations(self, date):
        for asset in self.assetsToUse:
            op = self.evaluateOperation(date, asset)
            if op == 'long' or op == 'short':
                self.pendingOperations.append(self.createPosition(asset, op, -1, -1, date))
            if op == 'skip':
                if self.verbose >= 2:
                    print('{} - Skipped {}, no profit predicted'.format(date.strftime('%Y-%m-%d'), asset))

    def evaluateOperation(self, date, asset):
        if np.isnan(self.dailyData[asset]['Close'][date]) or np.isnan(self.dailyData[asset]['Open'][date]):
            op = 'skip'
        elif self.predictedValues[asset][date] > self.dailyData[asset]['Close'][:date][-2]:# + 0.05:
            op = 'long'
        elif self.predictedValues[asset][date] < self.dailyData[asset]['Close'][:date][-2]:# - 0.05:
            op = 'short'
        else:
            op = 'skip'
        if self.verbose >= 2:
            print('{} - Close price at {}: {:.2f}. Predicted at {}: {:.2f}. Decided operation: {}'.format(asset, self.dailyData[asset]['Close'][:date].index[-2].strftime('%Y-%m-%d'), self.dailyData[asset]['Close'][:date][-2], date.strftime('%Y-%m-%d'), self.predictedValues[asset][date], op))
        return op

    def simulateOrders(self, start, end, simulationName = 'orders'):
        self.history[simulationName] = pd.DataFrame(index=self.dailyData[self.assetsToUse[0]][start:end].index, columns=['portfolioValue'])
        maxValue = self.funds / len(self.assetsToUse)
        date = self.dailyData[self.assetsToUse[0]][start:end].index[0]
        print('Portfolio value at start: {:.2f} BRL'.format(self.funds))
        self.history[simulationName].at[date, 'portfolioValue'] = self.funds
        # simulate for every day in the simulation period
        for d in self.dailyData[self.assetsToUse[0]][start:end].index:
            date = d
            #self.history[simulationName].at[date, 'portfolioValue'] = self.evaluatePortfolio(date)
            self.decideOrder(date)
            self.executeOperations(maxValue)
            self.liquidateAll()
            maxValue = self.funds / len(self.assetsToUse)
            self.history[simulationName].at[date, 'portfolioValue'] = self.evaluatePortfolio(date)
            if self.verbose >= 1:
                print('Portfolio value at {} market close: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), self.history[simulationName]['portfolioValue'][date]))
        print('Portfolio value at end: {:.2f} BRL'.format(self.funds))
        self.calculateDrawdown(simulationName)

    def decideOrder(self, date):
        for asset in self.assetsToUse:
            op = self.evaluateOrder(date, asset)
            if op == 'long' or op == 'short':
                self.pendingOperations.append(self.createPosition(asset, op, -1, -1, date))
            if op == 'skip':
                if self.verbose >= 2:
                    print('{} - Skipped {}, no profit predicted'.format(date.strftime('%Y-%m-%d'), asset))

    def evaluateOrder(self, date, asset):
        if np.isnan(self.dailyData[asset]['Close'][date]) or np.isnan(self.dailyData[asset]['Open'][date]):
            op = 'skip'
        elif (self.orders[asset][date] > self.confidenceLimit):
            op = 'long'
        elif (self.orders[asset][date] < (-self.confidenceLimit)):
            op = 'short'
        else:
            op = 'skip'
        if self.verbose >= 2:
            print('{} - Close price at {}: {:.2f}. Predicted movement at {}: {:.4f}. Decided operation: {}'.format(asset, self.dailyData[asset]['Close'][:date].index[-2].strftime('%Y-%m-%d'), self.dailyData[asset]['Close'][:date][-2], date.strftime('%Y-%m-%d'), self.orders[asset][date], op))
        return op

    def executeOperations(self, maxValue):
        for i in range(len(self.pendingOperations)):
            op = self.pendingOperations.pop(0)
            if op['opType'] == 'long':
                self.buy(op['asset'], op['date'], limitValue = maxValue)
            elif op['opType'] == 'short':
                self.sell(op['asset'], op['date'], limitValue = maxValue)

    def liquidateAll(self, date = None):
        for i in range(len(self.openPositions)):
            pos = self.openPositions[0] # always 0 because entries will be popped by buy or sell operations
            if pos['opType'] == 'long':
                self.sell(pos['asset'], date = date)
            elif pos['opType'] == 'short':
                self.buy(pos['asset'], date = date)

    def buy(self, asset, date = None, volume = None, limitValue = None):
        # check if a short operation with this asset exists, liquidate if so
        if len(filter(lambda op: op['asset'] == asset and op['opType'] == 'short', self.openPositions)) > 0:
            operation = self.openPositions.pop(findIndex(self.openPositions, asset, lambda x, y: x['asset'] == y))
            date = date if date else operation['date']
            price = self.dailyData[asset]['Close'][date] - 0.01
            buyValue = operation['volume'] * price
            fees = self.brokerage + (buyValue * self.transactionFees) + (self.brokerage * self.ISStax)
            self.funds = self.funds - buyValue - fees
            if self.verbose >= 2:
                print('{} Close - Bought {} {}. price: {} BRL, fees: {:.2f}, total: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), operation['volume'], asset, price, fees, buyValue + fees))
        else: #longs the stock
            price = self.dailyData[asset]['Open'][date] + 0.01
            if not volume and not limitValue:
                print('{} - Error buying {} - Neither volume nor limitValue were specified'.format(date, asset))
            else:
                limitVolume = int(((limitValue/price)//100)*100)
                volume = volume if volume else limitVolume
                buyValue = volume * price
                fees = self.brokerage + (buyValue * self.transactionFees) + (self.brokerage * self.ISStax)
                self.openPositions.append(self.createPosition(asset, 'long', volume, price, date))
                self.funds = self.funds - buyValue - fees
                if self.verbose >= 2:
                    print('{} Open -  Bought {} {}. price: {} BRL, fees: {:.2f}, total: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), volume, asset, price, fees, buyValue + fees))

    def sell(self, asset, date = None, volume = None, limitValue = None):
        # check if a long operation with this asset exists, liquidate if so
        if len(filter(lambda op: op['asset'] == asset and op['opType'] == 'long', self.openPositions)) > 0:
            operation = self.openPositions.pop(findIndex(self.openPositions, asset, lambda x, y: x['asset'] == y))
            date = date if date else operation['date']
            price = self.dailyData[asset]['Close'][date] - 0.01
            sellValue = operation['volume'] * price
            fees = self.brokerage + (sellValue * self.transactionFees) + (self.brokerage * self.ISStax)
            self.funds = self.funds + sellValue - fees
            if self.verbose >= 2:
                print('{} Close -  Sold  {} {}. price: {} BRL, fees: {:.2f}, total: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), operation['volume'], asset, price, fees, sellValue + fees))
        else: # shorts the stock
            price = self.dailyData[asset]['Open'][date] + 0.01
            if not volume and not limitValue:
                print('{} - Error shorting {} - Neither volume nor limitValue were specified'.format(date, asset))
            else:
                limitVolume = int(((limitValue/price)//100)*100)
                volume = volume if volume else limitVolume
                sellValue = volume * price
                fees = self.brokerage + (sellValue * self.transactionFees) + (self.brokerage * self.ISStax)
                self.openPositions.append(self.createPosition(asset, 'short', volume, price, date))
                self.funds = self.funds + sellValue - fees
                if self.verbose >= 2:
                    print('{} Open - Shorted {} {}. price: {} BRL, fees: {:.2f}, total: {:.2f} BRL'.format(date.strftime('%Y-%m-%d'), volume, asset, price, fees, sellValue + fees))

    def createPosition(self, asset, opType, volume, price, date = None):
        return {'asset': asset, 'opType': opType, 'volume': volume, 'price': price, 'date': date}

    def evaluatePortfolio(self, date, moment = 'Close'):
        if self.verbose >= 3: print('[-] PORTFOLIO CALCULATION ({}):'.format(date.strftime('%Y-%m-%d')))
        if self.verbose >= 3: print('Funds_________________________:{}'.format(self.funds))
        if self.verbose >= 3: print('Shorted Funds_________________:{}'.format(self.shortedFunds))
        portfolio = self.funds + self.shortedFunds
        for l in filter(lambda x: x['opType'] == 'long', self.openPositions):
            if self.verbose >= 3: print('{} Long____________________:{}'.format(l['asset'], l['volume'] * self.dailyData[l['asset']][moment][date]))
            portfolio += l['volume'] * self.dailyData[l['asset']][moment][date]
        for s in filter(lambda x: x['opType'] == 'short', self.openPositions):
            if self.verbose >= 3: print('{} Short___________________:{}'.format(s['asset'], -s['volume'] * self.dailyData[l['asset']][moment][date]))
            portfolio -= s['volume'] * self.dailyData[s['asset']][moment][date]
        return portfolio

    def calculateDrawdown(self, simulationName):
        simulationName = simulationName if simulationName else self.strategy
        if self.history[simulationName] is not None and self.history[simulationName]['portfolioValue'] is not None:
            drawdown = (self.history[simulationName]['portfolioValue'] - np.maximum.accumulate(self.history[simulationName]['portfolioValue']))/np.maximum.accumulate(self.history[simulationName]['portfolioValue'])
            self.history[simulationName] = self.history[simulationName].assign(drawdown = drawdown)

    def plotSimulations(self, simulations = None, names = None, initialPlotDate = None, finalPlotDate = None, locale = 'en', figsize = (20,6), legendPos = (-1., -0.4), legendncol = 1, legendsize = 15, linestyle = '-', linewidth = 4.0, saveImg = False, saveDir = '', saveName = ''):
        if len(self.history) == 0:
            print('No saved simulations found!')
            return None
        initialPlotDate = self.history[self.history.keys()[0]]['portfolioValue'][initialPlotDate:].index[0]
        finalPlotDate = self.history[self.history.keys()[0]]['portfolioValue'][:finalPlotDate].index[-1]

        fig, ax = plt.subplots(figsize=figsize, nrows = 1, ncols = 2)
        fig, ax[0] = self.plotPortfolio(simulations, names, initialPlotDate = initialPlotDate, finalPlotDate = finalPlotDate, locale = locale, figsize = (figsize[0], figsize[1]/2), linewidth = linewidth, titlePos = 'ax', fig = fig, ax = ax[0])
        fig, ax[1] = self.plotDrawdown(simulations, names, initialPlotDate = initialPlotDate, finalPlotDate = finalPlotDate, locale = locale, figsize = (figsize[0], figsize[1]/2), linewidth = linewidth, titlePos = 'ax', fig = fig, ax = ax[1])
        #plt.figlegend(bbox_to_anchor=(0., 0.), loc=3, ncol=legendncol, borderaxespad=0., prop={'size': legendsize}, frameon=False)
        plt.legend(bbox_to_anchor=legendPos, loc=3, ncol=legendncol, borderaxespad=0., prop={'size': legendsize}, frameon=False)
        if saveImg:
            saveName = saveName if saveName else 'unamed_backtest'
            fig.savefig('{}/{}.png'.format(saveDir, saveName), bbox_inches='tight')
            fig.savefig('{}/{}.pdf'.format(saveDir, saveName), bbox_inches='tight')
        return fig, ax

    def plotPortfolio(self, simulations = None, names = None, title = None, ylabel = None, initialPlotDate = None, finalPlotDate = None, locale = 'en', figsize = (10,6), legendsize = 15, linestyle = '-', linewidth = 4.0, legendncol = 1, titlePos = 'fig', saveImg = False, saveDir = '', saveName = '', fig = None, ax = None):
        if len(self.history) == 0:
            print('No saved simulations found!')
            return None
        initialPlotDate = self.history[self.history.keys()[0]]['portfolioValue'][initialPlotDate:].index[0]
        finalPlotDate = self.history[self.history.keys()[0]]['portfolioValue'][:finalPlotDate].index[-1]
        if not title:
            title = u'Portfolio Value' if locale != 'pt' else u'Valor do Portfólio'
        if not ylabel:
            ylabel = 'BRL'
        xLabel = 'Date' if locale != 'pt' else 'Date'

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize, nrows = 1, ncols = 1)
        if titlePos == 'fig':
            fig.suptitle(title)
        else:
            ax.set_title(title)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(ylabel)
        simulationsToPlot = len(simulations) if simulations is not None else len(self.history)
        simulations = simulations if simulations is not None else list(self.history.keys())
        if names is None:
            names = simulations
        formatter = EngFormatter(unit='')
        ax.yaxis.set_major_formatter(formatter)
        for i in range(simulationsToPlot):
            s = self.history[simulations[i]]['portfolioValue']
            d = pd.date_range(start=s[initialPlotDate:finalPlotDate].index[0], end=s[initialPlotDate:finalPlotDate].index[-1], freq="B")
            ax.plot(np.arange(len(s[initialPlotDate:finalPlotDate])), s[initialPlotDate:finalPlotDate], linestyle, label = names[i], linewidth = linewidth)
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            xticklabels = [(d[0] + x).strftime('%Y-%m') for x in xticks.astype(int)]
            ax.set_xticklabels(xticklabels)
        ax.plot(np.arange(len(s[initialPlotDate:finalPlotDate])), self.initialFunds * np.ones(np.arange(len(s[initialPlotDate:finalPlotDate])).shape), 'k--', linewidth=linewidth)
        ax.autoscale(True, axis='x')
        fig.autofmt_xdate()
        ax.grid()
        if legendncol > 1:
            plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3, ncol=legendncol, mode="expand", borderaxespad=0., prop={'size': legendsize}, frameon=False)
        else:
            plt.legend(bbox_to_anchor=(1.025, 1.), loc=2, ncol=1, borderaxespad=0., prop={'size': legendsize}, frameon=False)
        if saveImg:
            saveName = saveName if saveName else '{}'.format(s[0].name)
            fig.savefig('{}/{}.png'.format(saveDir, saveName), bbox_inches='tight')
            fig.savefig('{}/{}.pdf'.format(saveDir, saveName), bbox_inches='tight')
        return fig, ax

    def plotDrawdown(self, simulations = None, names = None, title = None, ylabel = None, initialPlotDate = None, finalPlotDate = None, locale = 'en', figsize = (10,6), legendsize = 15, linestyle = '-', linewidth = 4.0, legendncol = 1, titlePos = 'fig', saveImg = False, saveDir = '', saveName = '', fig = None, ax = None):
        if len(self.history) == 0:
            print('No saved simulations found!')
            return None
        initialPlotDate = self.history[self.history.keys()[0]]['portfolioValue'][initialPlotDate:].index[0]
        finalPlotDate = self.history[self.history.keys()[0]]['portfolioValue'][:finalPlotDate].index[-1]
        if not title:
            title = 'Drawdown'
        if not ylabel:
            ylabel = '%'
        xLabel = 'Date' if locale != 'pt' else 'Date'

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize, nrows = 1, ncols = 1)
        if titlePos == 'fig':
            fig.suptitle(title)
        else:
            ax.set_title(title)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(ylabel)
        simulationsToPlot = len(simulations) if simulations is not None else len(self.history)
        simulations = simulations if simulations is not None else list(self.history.keys())
        if names is None:
            names = simulations

        for i in range(simulationsToPlot):
            s = 100*self.history[simulations[i]]['drawdown']
            d = pd.date_range(start=s[initialPlotDate:finalPlotDate].index[0], end=s[initialPlotDate:finalPlotDate].index[-1], freq="B")
            ax.plot(np.arange(len(s[initialPlotDate:finalPlotDate])), s[initialPlotDate:finalPlotDate], linestyle, label = names[i], linewidth = linewidth)
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            xticklabels = [(d[0] + x).strftime('%Y-%m') for x in xticks.astype(int)]
            ax.set_xticklabels(xticklabels)
        ax.autoscale(True, axis='x')
        fig.autofmt_xdate()
        ax.grid()
        if legendncol > 1:
            plt.legend(bbox_to_anchor=(0., 1.00, 1., .102), loc=3, ncol=legendncol, mode="expand", borderaxespad=0., prop={'size': legendsize}, frameon=False)
        else:
            plt.legend(bbox_to_anchor=(1.025, 1.), loc=2, ncol=1, borderaxespad=0., prop={'size': legendsize}, frameon=False)
        if saveImg:
            saveName = saveName if saveName else '{}'.format(s[0].name)
            fig.savefig('{}/{}.png'.format(saveDir, saveName), bbox_inches='tight')
            fig.savefig('{}/{}.pdf'.format(saveDir, saveName), bbox_inches='tight')
        return fig, ax

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
