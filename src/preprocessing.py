# <editor-fold> IMPORTS
import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from datetime import date, datetime, timedelta
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import periodogram, adfuller, acf, pacf
from math import isnan
from pyTaLib.indicators import *
from neuralstocks.plots import *
from __future__ import print_function
%matplotlib inline
# </editor-fold>

# <editor-fold> FUNCTIONS DEF
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

def acquireData(replicateForHolidays = False, meanStdLen = None, returnCalcParams = [], SMAcols = [], SMAparams = [], EMAcols = [], EMAparams = [], dropNan = False):
    """
    Data Acquisition

    Parameters
    ----------
    replicateForHolidays : bool, indicates wheter or not to insert holidays in the
        series, replicating the last value. Every missing day that is not weekend
        is considered to be a holiday. Default False

    meanStdLen: int, length of rolling mean and standart deviation calculation.
        Default None

    returnCalcParams : array-like, array containing one or two-sized string arrays,
        refering to the columns used to calculate returns. If the array contains a
        single column, calculates log returns between actual and previos samples.
        If the array specifies 2 columns, calculates log return between corresponding
        samples of both columns. Default empty

    SMAparams, EMAparams : tuple (string, int) array-like, set of desired columns to
        calculate moving averages from, together with specific MA length. Default empty

    dropNan: bool, indicates either or not to exclude any lines containing nan-values
        of the dataSet. Default False

    Returns
    ----------
    df : pandas.DataFrame, DataFrame containing original data and any aditional
        calculations specified in function params
    """
    filepath = dataPath + '/' + assetType + '/' + asset + '/' + frequency + '/' + asset + '.CSV'
    df = pd.read_csv(filepath, delimiter=';', decimal=',',
                     parse_dates=['Date'], dayfirst=True, index_col='Date')
    df = df.sort_index() #csv entries begin from most recent to older dates

    if replicateForHolidays:
        df = insertHolidays(df)

    if meanStdLen:
        df = pd.concat([df, pd.Series(df['Close'].rolling(window=meanStdLen,center=False).mean(), name = 'Close_rollMean{}'.format(meanStdLen))], axis=1)
        df = pd.concat([df, pd.Series(df['Close'].rolling(window=meanStdLen,center=False).std(), name = 'Close_rollStd{}'.format(meanStdLen))], axis=1)

    for cols in returnCalcParams:
        if len(cols) == 1:
            df = pd.concat([df, logReturns(df[cols[0]])], axis=1)
        elif len(cols) == 2:
            df = pd.concat([df, logReturns(df[cols[0]], df[cols[1]])], axis=1)

    if len(SMAparams) > 0:
        for param in SMAparams:
            df = calculateSMAs(df, param[0], param[1])

    if len(EMAparams) > 0:
        for param in EMAparams:
            df = calculateEMAs(df, param[0], param[1])

    return df.dropna() if dropNan else df

def deTrend(df, column, window, model = 'additive', fitOrder = 1, weightModel = None, weightModelWindow = None,
            plot = False, initialPlotDate = None, finalPlotDate = None, overlap = False, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    if window < fitOrder + 1:
        window = fitOrder +1
        print('Warning: window must be at least {} samples wide for a fit of order {}. Adjusting window for minimal value.'.format(fitOrder+1, fitOrder))
    trendName = column + '_trend'
    residName = column + '_resid'
    weights = None

    if weightModel == 'full_pgram':
        weights = list(reversed(periodogram(df[column].dropna())[1 : window + 1]))
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif weightModel == 'full_acorr':
        weights = list(reversed(np.abs(acf(df[column].dropna(), nlags= window + 1))[1 : window + 1]))
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif weightModel == 'window_pgram':
        weightModelWindow = weightModelWindow if weightModelWindow or (not weightModelWindow >= 2 * window) else 2 * window
    elif weightModel == 'window_acorr':
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif not weightModel:
        weightModelWindow = window

    df[trendName] = np.empty(len(df[column]))*np.nan
    x = range(0, window)
    for i in range(0, len(df[column])):
        if i <= weightModelWindow:
            df[trendName].iloc[i] = np.nan
        else:
            if weightModel == 'window_pgram':
                weights = list(reversed(periodogram(df[column][i - weightModelWindow - 1 : i].dropna())[1 : window + 1]))
                if not checkIfTwoOrMoreValuesAreNotZero(weights):
                    weights = None
            elif weightModel == 'window_acorr':
                weights = list(reversed(np.abs(acf(df[column][i - weightModelWindow - 1: i].dropna(), nlags= window + 1))[1 : window + 1]))
                if np.isnan(weights).all() or not checkIfTwoOrMoreValuesAreNotZero(weights):
                    weights = None
            y = df[column][(i - window):i].values
            # print 'i: {}, xlen: {}, ylen: {}, window: {}, weightModelWindow: {}, weights: {}'.format(i, len(x), len(y), window, weightModelWindow, weights)
            a = np.polyfit(x, y, deg = fitOrder, w = weights);
            prediction = 0
            for j in range(fitOrder, -1, -1):
                prediction += a[fitOrder - j]*(window**j)
            df.set_value(df.index[i], trendName, prediction)

    df[residName] = df[column] / df[trendName] if model == 'multiplicative' else df[column] - df[trendName]

    if plot:
        initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
        finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
        fig, ax = plt.subplots(figsize=(15,10), nrows = 2 + int(not overlap), ncols = 1, sharex = True)
        plt.xlabel('Date')
        title = 'Observed and Predicted' if overlap else 'Observed'
        ax[0].set_title(title)
        ax[0].plot(df[column][initialPlotDate:finalPlotDate])
        if overlap:
            ax[0].plot(df[trendName][initialPlotDate:finalPlotDate], 'r')
            ax[0].legend()
        else:
            ax[1].set_title('Trend Estimation')
            ax[1].plot(df[trendName][initialPlotDate:finalPlotDate])
        signal = '/' if model.startswith('m') else '-'
        ax[1 + int(not overlap)].set_title('Observed {} Trend'.format(signal))
        ax[1 + int(not overlap)].plot(df[residName][initialPlotDate:finalPlotDate])

        plt.figtext(0.1,  0.010, 'deTrend Parameters', size = 14, verticalalignment = 'center')
        plt.figtext(0.1, -0.025, 'Model: {}'.format(model), size = 14)
        plt.figtext(0.1, -0.050, 'Window size: {}'.format(window), size = 14)
        plt.figtext(0.1, -0.075, 'Weight model: {:}'.format(weightModel), size = 14)
        if saveImg:
            fig.savefig('{}/deTrend_result{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

def checkIfTwoOrMoreValuesAreNotZero(x):
    notZero = False
    for i in range(len(x)):
        if notZero and x[i] != 0:
            return True
        elif x[i] != 0:
            notZero = True
    return False

def deSeason(df, column, freq, model = 'additive',
             plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveIndex = ''):
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
            fig.savefig('{}/deSeason_result{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

def seasonalMean(s, freq):
    return np.array([pd_nanmean(s[i::freq]) for i in range(freq)])

def decompose(df, column, model = 'additive', window = 3, fitOrder = 1, freq = 5,
              plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    trendName = column + '_trend'
    seasonalName = column + '_seasonal'
    residName = column + '_resid'
    df[trendName] = np.empty(len(df[column]))*np.nan
    df[seasonalName] = np.empty(len(df[column]))*np.nan
    df[residName] = np.empty(len(df[column]))*np.nan

    deTrend(df, column, window, model, fitOrder)
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
            fig.savefig('{}/decompose{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

def deTrendRMSE(df, column, model = 'additive', fitOrder = 1, windowMaxSize = 30, weights = None, weightModel = None, weightModelWindow = None, saveImg = False, saveIndex = ''):
    df2 = df.copy()
    model = 'multiplicative' if model.startswith('m') else 'additive'
    RMSE = np.empty(windowMaxSize + 1)*np.nan
    for i in range(fitOrder + 1, windowMaxSize + 1):
        deTrend(df2, column = column, window = i, model = model, fitOrder = fitOrder, weights = weights, weightModel = weightModel, weightModelWindow = weightModelWindow)
        if model == 'multiplicative':
            RMSE[i] = (np.square((df2['{}_resid'.format(column)].dropna() - 1)).sum())/(len(df2.dropna()))
        else:
            RMSE[i] = (np.square(df2['{}_resid'.format(column)].dropna()).sum())/(len(df2.dropna()))
    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1, sharex = True)
    ax.set_title('DeTrend RMSE per window size ({} model)'.format(model), fontsize = 20, fontweight = 'bold')
    ax.set_xlabel('Window size')
    ax.set_ylabel('RMSE')
    ax.plot(range(0,windowMaxSize+1), RMSE, 'bo')
    minValue = min(RMSE[fitOrder + 1 : windowMaxSize + 1])
    for i in range(fitOrder + 1, windowMaxSize + 1):
        if RMSE[i] == minValue:
            minIndex = i
    plt.annotate('local min', size = 18, xy=(minIndex, minValue), xytext=(minIndex*1.1, minValue*1.1), arrowprops=dict(facecolor='black', shrink=0.05))
    if saveImg:
        fig.savefig('{}/deTrend_RMSE{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

def deSeasonRMSE(df, column, model = 'additive', maxFreq = 20, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    df2 = df.copy()
    RMSE = np.empty(maxFreq + 1)*np.nan
    for i in range(0, maxFreq + 1):
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
        fig.savefig('{}/deSeason_RMSE{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

def testStationarity(ts, window, initialPlotDate='', finalPlotDate='', saveImg = False, saveIndex = ''):
    initialPlotDate = initialPlotDate if initialPlotDate else ts.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else ts.index[-1]

    #Determing rolling statistics
    rolmean = ts.dropna().rolling(window=window,center=False).mean()
    rolstd = ts.dropna().rolling(window=window,center=False).std()

    fig, ax = plt.subplots(figsize=(15,10), nrows = 1, ncols = 1, sharex = True)
    #Plot rolling statistics:
    ax.plot(ts[initialPlotDate:finalPlotDate], color='blue',label='Original')
    ax.plot(rolmean[initialPlotDate:finalPlotDate], color='red', label='Rolling Mean')
    ax.plot(rolstd[initialPlotDate:finalPlotDate], color='black', label = 'Rolling Std')
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Standard Deviation')

    #Perform Dickey-Fuller test:
    #print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(ts.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print dfoutput

    plt.figtext(0.1,  0.010, 'Results of Dickey-Fuller Test:', size = 14, verticalalignment = 'center')
    plt.figtext(0.1, -0.025, 'Test Statistic {:48.6f}'.format(dfoutput['Test Statistic']), size = 14)
    plt.figtext(0.1, -0.050, 'p-value {:58.6f}'.format(dfoutput['p-value']), size = 14)
    plt.figtext(0.1, -0.075, '#Lags Used {:51.6f}'.format(dfoutput['#Lags Used']), size = 14)
    plt.figtext(0.1, -0.100, 'Number of Observations Used {:20.6f}'.format(dfoutput['Number of Observations Used']), size = 14)
    plt.figtext(0.1, -0.125, 'Critical Value (1%) {:41.6f}'.format(dfoutput['Critical Value (1%)']), size = 14)
    plt.figtext(0.1, -0.150, 'Critical Value (5%) {:41.6f}'.format(dfoutput['Critical Value (5%)']), size = 14)
    plt.figtext(0.1, -0.175, 'Critical Value (10%) {:39.6f}'.format(dfoutput['Critical Value (10%)']), size = 14)

    if saveImg:
        fig.savefig('{}/testStationarity{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

# </editor-fold>

# <editor-fold> GLOBAL PARAMS
dataPath = '/home/danilofrp/projeto_final/data'
assetType = 'stocks'
asset = 'PETR4'
frequency = 'diario'

decomposeModel = 'additive'

saveDir = '/home/danilofrp/projeto_final/results/preprocessing/misc'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
saveFormat = 'png'

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
# </editor-fold>

# <editor-fold> workspace
df = acquireData(replicateForHolidays = True,
                 meanStdLen = 25,
                 returnCalcParams = [['Close'], ['Close', 'Open']],
                 EMAparams = [('Close', 17), ('Close', 72), ('Volume', 21)],
                 dropNan = False)
df.tail(10)

plotSeries(df['Close'], asset = asset,  initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotReturnSeries(df, column = 'Close', asset = asset,  initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = None,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weightModel = 'full_pgram',
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = 'window_pgram',
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 10, model = decomposeModel, fitOrder = 1, weightModel = 'window_acorr',
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deSeason(df, column = 'Close', freq = 5, model = decomposeModel, plot = True, initialPlotDate = '2017', finalPlotDate = '2017')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 10, weightModel = None, saveImg = False, saveIndex = '')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weightModel = 'full_pgram', saveImg = False, saveIndex = '')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weightModel = 'window_pgram', saveImg = False, saveIndex = '')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weightModel = 'window_acorr', saveImg = False, saveIndex = '')

deSeasonRMSE(df, column = 'Close', model = decomposeModel, maxFreq = 75, saveImg = False, saveIndex = '')

decompose(df, column = 'Close', model = decomposeModel, window = 3, freq = 5,
          plot = False, initialPlotDate = '2008', finalPlotDate = '2008')

plotPeriodogramStats(df['Close_EMA72_logdiff'], plotInit = 2, plotEnd = 100, yLog = False, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotPeriodogramSciPy(df['Close_EMA72_logdiff'], plotInit = 2, plotEnd = 100, yLog = False, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotFFT(df['Close_EMA72_logdiff'], saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotSeasonalDecompose(df['Close'],  asset = asset, frequency=5, initialPlotDate='2016', finalPlotDate='2017', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

testStationarity(df['Close_resid'], window=20, initialPlotDate='', finalPlotDate='', saveImg = False, saveIndex = '')

plotAcf(df['Close'][:'2008'][-20:], lags = 40, partialAcf = False, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotCrosscorrelation(df['Close_returns'], df['Close_EMA72_logdiff'], 50, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

histogram([df['Close'], df['Close_trend']], colors = ['b', 'r'], nBins=100, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weights = None, weightModel = 'full_pgram', plot = False)
ax.plot(df['Close_trend'], df['Close'], 'bo')
deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weights = None, weightModel = 'window_pgram', plot = False)
ax.plot(df['Close_trend'], df['Close'], 'ro', alpha=0.5)

scatterHist(df['Close'], df['Close_trend'], nBins = 100, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

# </editor-fold>

# <editor-fold> ARIMA
model = ARIMA(df['Close_r'][:'2016'], order=(2, 1, 1))
results_ARIMA = model.fit(disp=-1)
fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
#ax.plot(results_ARIMA.resid['2016-07':'2016-12'])
ax.plot(df['Close_r']['2016-07':'2016-12'])
ax.plot(results_ARIMA.fittedvalues['2016-07':'2016-12'], color='red')
ax.axhline(y=0,linestyle='--',color='gray')
ax.set_title('RSS: %.4f'% sum((results_ARIMA.fittedvalues['2016-07':'2016-12']-df['Close_r']['2016-07':'2016-12'])**2))
#fig.savefig('{}/arima_fitted3.{}'.format(saveDir, saveFormat), bbox_inches='tight')


print(results_ARIMA.summary())
# plot residual errors
residuals = pd.DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(np.log(df['Close'].iloc[1]), index=df['Close'].index)
#print predictions_ARIMA_log
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
fig, ax = plt.subplots(figsize=(15,10), nrows = 1, ncols = 1, sharex = True)
ax.plot(df['Close'][:'2016'])
ax.plot(predictions_ARIMA[:'2016'])
ax.set_title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA[:'2016']-df['Close'][:'2016'])**2)/len(df['Close'][:'2016'])))
#fig.savefig('{}/close_fitted1.{}'.format(saveDir, saveFormat), bbox_inches='tight')

# </editor-fold>


# <editor-fold> MISC
df2 = df.copy()
windowMaxSize = 25
weightModelWindowMaxSize = 6
model = 'additive'
column = 'Close'
RMSE = np.empty((windowMaxSize + 1, weightModelWindowMaxSize + 1), dtype=float)*0
for i in range(2, windowMaxSize + 1):
    for j in range(1, weightModelWindowMaxSize + 1):
        print('Running deTrend ({}, {})'.format(i, j), end='\r')
        deTrend(df2, column = column, window = i, model = model, fitOrder = 1, weightModel = 'window_acorr', weightModelWindow = j * windowMaxSize)
        if model.startswith('m'):
            RMSE[i, j] = np.square((df2['{}_resid'.format(column)] - 1)).sum()/len(df2['{}_resid'.format(column)].dropna())
        else:
            RMSE[i, j] = np.square(df2['{}_resid'.format(column)]).sum()/len(df2['{}_resid'.format(column)].dropna())
        if i == 2 and j == 1:
            minimal = RMSE[i][j]
            iMin = i
            jMin = j
        if RMSE[i, j] < minimal:
            minimal = RMSE[i][j]
            iMin = i
            jMin = j

print('Minimal RSME = {}, at window = {} and weightModelWindow = {} * window'.format(minimal, iMin, jMin))

fig, ax = plt.subplots(figsize=(10,10))
# plt.imshow(RMSE[2:,1:], cmap="jet", extent=[0, weightModelWindowMaxSize, 1, windowMaxSize], aspect="auto")
cax = ax.imshow(RMSE[2:,1:], cmap="jet", aspect="auto")
plt.gca().invert_yaxis()
cbar = fig.colorbar(cax)



df3 = df.copy()
windowMaxSize = 25
weightModelWindowMaxSize = 6
model = 'additive'
column = 'Close'
RMSE3 = np.empty((windowMaxSize + 1, weightModelWindowMaxSize + 1), dtype=float)*0
for i in range(2, windowMaxSize + 1):
    for j in range(2, weightModelWindowMaxSize + 1):
        print('Running deTrend ({}, {})'.format(i, j), end='\r')
        deTrend(df3, column = column, window = i, model = model, fitOrder = 1, weightModel = 'window_pgram', weightModelWindow = j * windowMaxSize)
        if model.startswith('m'):
            RMSE3[i, j] = np.square((df3['{}_resid'.format(column)] - 1)).sum()/len(df3['{}_resid'.format(column)].dropna())
        else:
            RMSE3[i, j] = np.square(df3['{}_resid'.format(column)]).sum()/len(df3['{}_resid'.format(column)].dropna())
        if i == 2 and j == 2:
            minimal = RMSE3[i][j]
            iMin = i
            jMin = j
        if RMSE3[i, j] < minimal:
            minimal = RMSE3[i][j]
            iMin = i
            jMin = j

print('Minimal RSME = {}, at window = {} and weightModelWindow = {} * window'.format(minimal, iMin, jMin))

fig, ax = plt.subplots(figsize=(10,10))
# plt.imshow(RMSE3[2:,1:], cmap="jet", extent=[0, weightModelWindowMaxSize, 1, windowMaxSize], aspect="auto")
cax = ax.imshow(RMSE3[2:,2:], cmap="jet", aspect="auto")
plt.gca().invert_yaxis()
cbar = fig.colorbar(cax)



df2 = df.copy()
windowMaxSize = 20
maxFreq = 30
model = 'additive'
column = 'Close'
RSS = np.empty((windowMaxSize + 1, maxFreq + 1), dtype=float)*0
for i in range(2, windowMaxSize + 1):
    deTrend(df2, column = column, window = i, model = model, fitOrder = 1)
    for j in range(maxFreq + 1):
        deSeason(df2, column = column, freq = j, model = model)
        if model == 'multiplicative':
            RSS[i, j] = np.square((df2['{}_resid'.format(column)] - 1)).sum()
        else:
            RSS[i, j] = np.square(df2['{}_resid'.format(column)]).sum()

fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(RSS[2:,:], cmap="jet", extent=[2, windowMaxSize, 0, maxFreq], aspect="auto")
cbar = plt.colorbar()
# </editor-fold>
