from __future__ import print_function
import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pyTaLib.indicators import *
from neuralstocks.utils import *
from neuralstocks.plots import *

def insertHolidays(df):
    """
    Inserts Holidays in a time series, replicating the last value. Adds a
    'Holiday' column in the dataset, where 1 indicates a holiday and 0
    indicates an usual business day. This function does not contain a
    callendar, it treats every missing business as a holiday. This function
    does not insert weekends in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame object, dataframe in which to insert holidays. Indexes
    must be of type datetime.

    Returns
    ----------
    df : pandas.DataFrame object, original dataframe with holidays added and new
       'Holidays' column
    """
    start = df.index[0]
    end = df.index[-1]
    step = timedelta(days=1)

    df.loc[:,'Holiday'] = 0
    while start <= end:
        if start.weekday() < 5: # 5 - Saturday, 6 - Sunday
            if not (start in df.index):
                df.loc[start] = df[:start].iloc[-1]
                df.loc[start, 'Holiday'] = 1
                df.sort_index(inplace = True)
            else:
                df.loc[start, 'Holiday'] = 0
        start += step

    return df

def logReturns(s1, s2 = pd.Series([])):
    """
    Log Returns of time series

    Parameters
    ----------
    s1, s2 : pandas.Series objects if s2 is not defined, calculates log returns
    between actual and previous samples of s1

    Returns
    ----------
    log_returns : pandas.Series
    """
    columnName = s1.name + '_returns' if (s2.empty or s1.name == s2.name) else s1.name + '/' + s2.name + '_returns'
    if s2.empty or s1.name == s2.name:
        return pd.Series(np.log(s1/s1.shift()), name = columnName)
    else:
        return pd.Series(np.log(s1/s2), name = columnName)

def calculateSMAs(df, column, lenghts):
    """
    Simple Moving Averages calculation

    Parameters
    ----------
    df : pandas.DataFrame object
    lenghts : int array, set of desired lenghts to calculate moving averages

    Returns
    ----------
    df : pandas.DataFrame, original DataFrame concatenated with moving
        averages and log differences between column moving averages
    """
    if isinstance(lenghts, (int, long)):
        lenghts = [lenghts]
    for l in lenghts:
        sma = SMA(df, column, l)
        df = pd.concat([df, sma], axis = 1)
        df = pd.concat([df, pd.Series(logReturns(df[column], sma), name='{}_SMA{}_logdiff'.format(column, l))], axis = 1)
    return df

def calculateEMAs(df, column, lenghts):
    """
    Exponential Moving Averages calculation

    Parameters
    ----------
    df : pandas.DataFrame object
    column : string, column from which to calculate Moving Averages
    lenghts : int array, set of desired lenghts to calculate moving averages

    Returns
    ----------
    df : pandas.DataFrame, original DataFrame concatenated with moving
        averages and log differences between column and moving averages
    """
    if isinstance(lenghts, (int, long)):
        lenghts = [lenghts]
    for l in lenghts:
        ema = EMA(df, column, l)
        df = pd.concat([df, ema], axis = 1)
        df = pd.concat([df, pd.Series(logReturns(df[column], ema), name='{}_EMA{}_logdiff'.format(column, l))], axis = 1)
    return df

def deTrend(df, column, window, model = 'additive', fitOrder = 1, weightModel = None, weightModelWindow = None,
            plot = False, initialPlotDate = None, finalPlotDate = None, overlap = False, detailed = False,
            saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    if window < fitOrder + 1:
        window = fitOrder +1
        print('Warning: window must be at least {} samples wide for a fit of order {}. Adjusting window for minimal value.'.format(fitOrder+1, fitOrder))
    trendName = column + '_trend'
    residName = column + '_resid'
    weights = None

    weights, weightModelWindow = getWeights(df[column], window = window, weightModel = weightModel, weightModelWindow = weightModelWindow)

    df[trendName] = np.empty(len(df[column]))*np.nan
    for i in range(0, len(df[column])):
        if i <= weightModelWindow:
            df.set_value(df.index[i], trendName, np.nan)
        else:
            if weightModel and weightModel.startswith('window_'):
                weights, weightModelWindow = getWeights(df[column][i - weightModelWindow - 1 : i], window = window, weightModel = weightModel, weightModelWindow = weightModelWindow)
            df.set_value(df.index[i], trendName, predict(x = range(0, window), y = df[column][(i - window):i].values, fitOrder = fitOrder, weights = weights, window = window))

    df[residName] = df[column] / df[trendName] if model.startswith('m') else df[column] - df[trendName]
    RMSE = (np.square(df['{}_resid'.format(column)].dropna() - int(model.startswith('m'))).sum())/(len(df.dropna()))

    if plot:
        plotDeTrendResult(df = df, column = column, window = window, model = model, weightModel = weightModel, weightModelWindow = weightModelWindow, RMSE = RMSE,
                          initialPlotDate = initialPlotDate, finalPlotDate = finalPlotDate, overlap = overlap, detailed = detailed,
                          saveImg = saveImg, saveDir = saveDir, saveName = saveName, saveFormat = saveFormat)

def deSeason(df, column, freq, model = 'additive',
             plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    trendName = column + '_trend'
    seasonalName = column + '_seasonal'
    residName = column + '_resid'
    if freq > 0:
        if model == 'multiplicative':
            df[residName] = df[column] / df[trendName]
        else:
            df[residName] = df[column] - df[trendName]
        df[seasonalName] = np.empty(len(df[column]))*np.nan
        seasonalMeans = seasonalMean(df[residName], freq)
        for i in range(len(df[seasonalName])):
            df.set_value(df.index[i], seasonalName, seasonalMeans[i%freq])
        if model == 'multiplicative':
            df[residName] /= df[seasonalName]
        else:
            df[residName] -= df[seasonalName]
    else:
        if model == 'multiplicative':
            df[seasonalName] = pd.Series(1, df.index)
        else:
            df[seasonalName] = pd.Series(0, df.index)

    if plot:
        initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
        finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
        fig, ax = plt.subplots(figsize=(15,10), nrows = 4, ncols = 1, sharex = True)
        plt.xlabel('Date')
        ax[0].set_title('Observed')
        ax[0].plot(df[column][initialPlotDate:finalPlotDate])
        ax[1].set_title('Trend Component')
        ax[1].plot(df[trendName][initialPlotDate:finalPlotDate])
        ax[2].set_title('Seasonal Component')
        ax[2].plot(df[seasonalName][initialPlotDate:finalPlotDate])
        ax[3].set_title('Residuals ({} model)'.format(model))
        ax[3].plot(df[residName][initialPlotDate:finalPlotDate])
        if saveImg:
            saveName = saveName if saveName else '{}_deSeason'.format(s.name)
            fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def decompose(df, column, model = 'additive', window = 3, fitOrder = 1, freq = 5, weightModel = None, weightModelWindow = None,
              plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    trendName = column + '_trend'
    seasonalName = column + '_seasonal'
    residName = column + '_resid'
    df[trendName] = np.empty(len(df[column]))*np.nan
    df[seasonalName] = np.empty(len(df[column]))*np.nan
    df[residName] = np.empty(len(df[column]))*np.nan

    deTrend(df, column, window, model, fitOrder, weightModel, weightModelWindow)
    deSeason(df, column, freq, model)

    if plot:
        initialPlotDate = initialPlotDate if initialPlotDate else df[column].index[0]
        finalPlotDate = finalPlotDate if finalPlotDate else df[column].index[-1]
        title = asset + ' ' + column + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' ' + column + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'

        fig, ax = plt.subplots(figsize=(10,15), nrows = 4, ncols = 1)

        plt.xlabel('Date')
        ax[0].set_title(title)
        ax[0].plot(df[initialPlotDate:finalPlotDate].index, df[column][initialPlotDate:finalPlotDate])
        ax[0].grid()

        ax[1].set_title('Trend')
        ax[1].plot(df[initialPlotDate:finalPlotDate].index,df[trendName][initialPlotDate:finalPlotDate])
        ax[1].grid()

        ax[2].set_title('Seasonal')
        ax[2].plot(df[initialPlotDate:finalPlotDate].index,df[seasonalName][initialPlotDate:finalPlotDate])
        ax[2].grid()

        ax[3].set_title('Residuals')
        ax[3].plot(df[initialPlotDate:finalPlotDate].index,df[residName][initialPlotDate:finalPlotDate])
        ax[3].grid()

        if saveImg:
            saveName = saveName if saveName else '{}_decompose'.format(s.name)
            fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def deTrendRMSE(df, column, model = 'additive', fitOrder = 1, windowMaxSize = 30, weightModel = None, weightModelWindow = None,
                figsize = (10,10), saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    df2 = df.copy()
    model = 'multiplicative' if model.startswith('m') else 'additive'
    RMSE = np.empty(windowMaxSize + 1)*np.nan
    for i in range(fitOrder + 1, windowMaxSize + 1):
        print('Running deTrend ({})'.format(i), end='\r')
        deTrend(df2, column = column, window = i, model = model, fitOrder = fitOrder, weightModel = weightModel, weightModelWindow = weightModelWindow)
        if model == 'multiplicative':
            RMSE[i] = np.sqrt(np.square((df2['{}_resid'.format(column)].dropna() - 1)).sum()/(len(df2.dropna())))
        else:
            RMSE[i] = np.sqrt(np.square(df2['{}_resid'.format(column)].dropna()).sum()/(len(df2.dropna())))
    fig, ax = plt.subplots(figsize=figsize, nrows = 1, ncols = 1, sharex = True)
    ax.set_title('DeTrend RMSE per window size', fontsize = 20, fontweight = 'bold')
    ax.set_xlabel('Window size')
    ax.set_ylabel('RMSE')
    ax.grid()
    ax.plot(range(0,windowMaxSize+1), RMSE, 'bo', markersize=14)
    minValue = min(RMSE[fitOrder + 1 : windowMaxSize + 1])
    for i in range(fitOrder + 1, windowMaxSize + 1):
        if RMSE[i] == minValue:
            minIndex = i
    #plt.annotate('local min', size = 18, xy=(minIndex, minValue), xytext=(minIndex*1.1, minValue*1.1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.figtext(0.45,  -0.05, 'Minimal RMSE: {:.5f}, using Trend window of {} samples'.format(RMSE[minIndex], minIndex), size = 18, horizontalalignment = 'center')
    if saveImg:
        saveName = saveName if saveName else '{}_deTrendRMSE'.format(df[column].name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def deSeasonRMSE(df, column, model = 'additive', maxFreq = 20, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    df2 = df.copy()
    RMSE = np.empty(maxFreq + 1)*np.nan
    for i in range(0, maxFreq + 1):
        print('Running deSeason ({})'.format(i), end='\r')
        deSeason(df2, column = column, freq = i, model = model)
        if model == 'multiplicative':
            RMSE[i] = (np.square((df2['{}_resid'.format(column)] - 1)).sum())/(len(df2.dropna()))
        else:
            RMSE[i] = (np.square(df2['{}_resid'.format(column)]).sum())/(len(df2.dropna()))
    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1, sharex = True)
    ax.set_title('DeSeason RMSE per assumed period ({} model)'.format(model))
    ax.set_xlabel('Period (samples)')
    ax.set_ylabel('RMSE')

    ax.plot(range(0,maxFreq+1), RMSE, 'bo')
    minValue = min(RMSE[0 : maxFreq + 1])
    for i in range(0, maxFreq + 1):
        if RMSE[i] == minValue:
            minIndex = i
    plt.annotate('local min', size = 18, xy=(minIndex, minValue), xytext=(minIndex, minValue), arrowprops=dict(facecolor='black', shrink=0.05))
    if saveImg:
        saveName = saveName if saveName else '{}_deSeasonRMSE'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')
