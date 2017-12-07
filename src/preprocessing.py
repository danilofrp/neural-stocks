# <editor-fold> IMPORTS
import sys
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import entropy
from datetime import date, datetime, timedelta
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import periodogram, adfuller, acf, pacf
from math import isnan
from pyTaLib.indicators import *
%matplotlib inline
# </editor-fold>

# <editor-fold> FUNCTIONS DEF
def insertMissingDays(df):
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
    """Log Returns of time series
    Parameters
    ----------
    s1, s2 : pandas.Series objects
             if s2 is not defined, calculates log
             returns between actual and
             previous samples of s1

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
    """Simple Moving Averages calculation
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
    """Exponential Moving Averages calculation
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
    """Data Acquisition
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
        df = insertMissingDays(df)

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

def plotSeries(s, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = ''):
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0].strftime('%d-%m-%Y')
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1].strftime('%d-%m-%Y')
    title = '{} {} ({})'.format(asset, s.name, initialPlotDate) if initialPlotDate == finalPlotDate else '{} {} ({} to {})'.format(asset, s.name, initialPlotDate, finalPlotDate)

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1)
    fig.suptitle(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.plot(s[initialPlotDate:finalPlotDate])
    if saveImg:
        fig.savefig('{}/{}{}.{}'.format(saveImgFolder, column, saveIndex, saveImgFormat), bbox_inches='tight')

def plotReturnSeries(df, column, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = ''):
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    title = asset + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'
    returnName = column + '_returns'

    fig, ax = plt.subplots(figsize=(10,10), nrows = 2, ncols = 1, sharex = True)

    plt.xlabel('Date')
    plt.title(title)
    ax[0].set_ylabel('Price')
    ax[0].plot(df[column][initialPlotDate:finalPlotDate])
    ax[0].grid()

    ax[1].set_ylabel('Returns')
    ax[1].plot(df[returnName][initialPlotDate:finalPlotDate])
    ax[1].grid()

    if saveImg:
        fig.savefig('{}/returns{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def deTrend(df, column, window, model = 'additive', fitOrder = 1, weights = None, weightModel = None,
            plot = False, initialPlotDate = None, finalPlotDate = None, overlap = False, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    if window < fitOrder + 1:
        window = fitOrder +1
        print 'Warning: window must be at least {} samples wide for a fit of order {}. Adjusting window for minimal value.'.format(fitOrder+1, fitOrder)
    trendName = column + '_trend'
    residName = column + '_resid'

    if ((not weights) and weightModel):
        if weightModel == 'periodogram':
            weights = list(reversed(periodogram(df[column].dropna())[1 : window + 1]))
        elif weightModel == 'autocorrelogram':
            weights = list(reversed(acf(df[column].dropna(), nlags= window + 1)[1 : window + 1]))

    df[trendName] = np.empty(len(df[column]))*np.nan
    x = range(0, window)
    for i in range(0, len(df[column])):
        if i < (window):
            df[trendName].iloc[i] = np.nan
        else:
            if weightModel == 'adaptative_pgram' and i >= window * 2:
                weights = list(reversed(periodogram(df[column][:i].dropna())[1 : window + 1]))
            y = df[column][(i - window):i].values
            a = np.polyfit(x, y, deg = fitOrder, w = weights)
            prediction = 0
            for j in range(fitOrder, -1, -1):
                prediction += a[fitOrder - j]*(window**j)
            df.set_value(df.index[i], trendName, prediction)
    if model == 'multiplicative':
        df[residName] = df[column] / df[trendName]
    else :
        df[residName] = df[column] - df[trendName]

    if plot:
        initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
        finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
        rows = 2 + int(not overlap)
        fig, ax = plt.subplots(figsize=(15,10), nrows = rows, ncols = 1, sharex = True)
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
        plt.figtext(0.1, -0.050, 'Window: {}'.format(window), size = 14)
        plt.figtext(0.1, -0.075, 'Weight Model: {:}'.format(weightModel), size = 14)
        if saveImg:
            fig.savefig('{}/deTrend_result{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

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
        seasonalMeans = seasonalMean(df, residName, freq)
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
            fig.savefig('{}/deSeason_result{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def seasonalMean(df, column, freq):
    return np.array([pd_nanmean(df[column][i::freq]) for i in range(freq)])

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
            fig.savefig('{}/decompose{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def deTrendRMSE(df, column, model = 'additive', fitOrder = 1, windowMaxSize = 30, weights = None, weightModel = None, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    df2 = df.copy()
    RMSE = np.empty(windowMaxSize + 1)*np.nan
    for i in range(fitOrder + 1, windowMaxSize + 1):
        deTrend(df2, column = column, window = i, model = model, fitOrder = fitOrder, weights = weights, weightModel = weightModel)
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
        fig.savefig('{}/deTrend_RMSE{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def deSeasonRMSE(df, column, model = 'additive', maxFreq = 20, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    df2 = df.copy()
    nanSamples = 0
    for i in range(len(df2)):
        if not np.isnan(df2['{}_resid'.format(column)].iloc[i]):
            nanSamples = i
            break
    RMSE = np.empty(maxFreq + 1)*np.nan
    for i in range(0, maxFreq + 1):
        deSeason(df2, column = column, freq = i, model = model)
        if model == 'multiplicative':
            RMSE[i] = (np.square((df2['{}_resid'.format(column)] - 1)).sum())/(len(df2) - nanSamples)
        else:
            RMSE[i] = (np.square(df2['{}_resid'.format(column)]).sum())/(len(df2) - nanSamples)
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
        fig.savefig('{}/deSeason_RMSE{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def plotPeriodogram(s, plotInit = 0, plotEnd = None, yLog = False, saveImg = False, saveIndex = ''):
    pgram = periodogram(s.dropna())
    plotEnd = plotEnd if plotEnd else len(s.dropna())/2

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
    fig.suptitle('Periodogram')
    ax.set_xlabel('Period (samples)')
    if yLog:
        plt.yscale('log')
    ax.stem(range(plotInit,plotEnd+1), pgram[plotInit:plotEnd+1])
    if saveImg:
        fig.savefig('{}/periodogram{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def FFT(s, saveImg = False, saveIndex = ''):
    Fs = 1.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    y = s.dropna() - 1 if s.dropna().mean() > 0.5 else s.dropna()
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]

    fig, ax = plt.subplots(2, 1, figsize=(15,10))
    ax[0].plot(y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq,abs(Y), 'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (1/sample)')
    ax[1].set_ylabel('|X(freq)|')
    if saveImg:
        fig.savefig('{}/fft{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def plotSeasonalDecompose(df, column, frequency = 1, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = ''):
    if isnan(df[column].iloc[0]):
        df = df.drop(df.index[0])
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    title = asset + ' ' + column + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' ' + column + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'
    initialIndex = np.where(df.index == df[initialPlotDate:finalPlotDate].index[0])[0][0]
    finalIndex = np.where(df.index == df[initialPlotDate:finalPlotDate].index[-1])[0][0] + 1

    result = seasonal_decompose(df[column].values, model='a', freq=frequency, two_sided=False)

    fig, ax = plt.subplots(figsize=(10,15), nrows = 4, ncols = 1)

    plot_data = df[initialPlotDate:finalPlotDate]
    plt.xlabel('Date')
    ax[0].set_title(title)
    ax[0].plot(df[initialPlotDate:finalPlotDate].index,plot_data[column],'b-')
    #ax[0].plot(df[initialPlotDate:finalPlotDate].index,plot_data['Open'],'r:')
    #ax[0].plot(df[initialPlotDate:finalPlotDate].index,plot_data['High'],'g:')
    #ax[0].plot(df[initialPlotDate:finalPlotDate].index,plot_data['Low'],'g:')
    ax[0].grid()

    ax[1].set_title('trend')
    ax[1].plot(df[initialPlotDate:finalPlotDate].index,result.trend[initialIndex:finalIndex])
    ax[1].grid()

    ax[2].set_title('seasonal')
    ax[2].plot(df[initialPlotDate:finalPlotDate].index,result.seasonal[initialIndex:finalIndex])
    ax[2].grid()

    ax[3].set_title('resid')
    ax[3].plot(df[initialPlotDate:finalPlotDate].index,result.resid[initialIndex:finalIndex])
    ax[3].grid()

    if saveImg:
        fig.savefig('{}/seasonal_decompose{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def testStationarity(ts, window, initialPlotDate='', finalPlotDate='', saveImg = False, saveIndex = ''):
    if isnan(ts.iloc[0]):
        ts = ts.drop(ts.index[0])
    initialPlotDate = initialPlotDate if initialPlotDate else ts.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else ts.index[-1]

    #Determing rolling statistics
    rolmean = ts.rolling(window=window,center=False).mean()
    rolstd = ts.rolling(window=window,center=False).std()

    fig, ax = plt.subplots(figsize=(15,10), nrows = 1, ncols = 1, sharex = True)
    #Plot rolling statistics:
    ax.plot(ts[initialPlotDate:finalPlotDate], color='blue',label='Original')
    ax.plot(rolmean[initialPlotDate:finalPlotDate], color='red', label='Rolling Mean')
    ax.plot(rolstd[initialPlotDate:finalPlotDate], color='black', label = 'Rolling Std')
    ax.legend(loc='best')
    ax.set_title('Rolling Mean & Standard Deviation')

    #Perform Dickey-Fuller test:
    #print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(ts, autolag='AIC')
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
        fig.savefig('{}/testStationarity{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def plotAcf(s, lags = 10, partialAcf = False, saveImg = False, saveIndex = ''):
    lag_acf = acf(s.dropna(), nlags=lags)
    if partialAcf:
        lag_pacf = pacf(s.dropna(), nlags=lags, method='ols')

    fig, ax = plt.subplots(figsize=(20 if partialAcf else 10,10), nrows = 1, ncols = 2 if partialAcf else 1)
    if not partialAcf:
        #Plot ACF:
        ax.set_title('Autocorrelation Function')
        ax.set_xlabel('Lags')
        ax.stem(range(1,len(lag_acf)),lag_acf[1:])
        ax.axhline(y=0,linestyle='--',color='gray')
        ax.axhline(y=-7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
        ax.axhline(y=7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
    else:
        #Plot ACF:
        ax[0].set_title('Autocorrelation Function')
        ax[0].set_xlabel('Lags')
        ax[0].stem(range(1,len(lag_acf)),lag_acf[1:])
        ax[0].axhline(y=0,linestyle='--',color='gray')
        ax[0].axhline(y=-7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
        ax[0].axhline(y=7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
        #Plot PACF:
        ax[1].set_title('Partial Autocorrelation Function')
        ax[1].set_xlabel('Lags')
        ax[1].stem(lag_pacf)
        ax[1].axhline(y=0,linestyle='--',color='gray')
        ax[1].axhline(y=-7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
        ax[1].axhline(y=7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')

    if saveImg:
        fig.savefig('{}/acf_pacf{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def crosscorrelation(x, y, nlags = 0):
    """Cross correlations calculatins until nlags.
    Parameters
    ----------
    nlags : int, number of lags to calculate cross-correlation, default 0
    x, y : pandas.Series objects of equal length

    Returns
    ----------
    crosscorrelation : [float]
    """
    return [x.corr(y.shift(lag)) for lag in range(nlags + 1)]

def plotCrosscorrelation(x, y, nlags = 10, saveImg = False, saveIndex = 0):
    """Cross correlations calculatins until nlags.
    Parameters
    ----------
    x, y : pandas.Series objects of equal length
    nlags : int, number of lags to calculate cross-correlation, default 10
    saveImg : bool, saves image to save directory if True, default False
    saveIndex: string, sufix to add to saved image file name, default empty
    """
    crosscorrelationelation = crosscorrelation(x, y, nlags)

    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1)
    #Plot ACF:
    ax.set_title('crosscorrelationelation ({} and {})'.format(x.name, y.name))
    ax.set_xlabel('Lags')
    ax.stem(crosscorrelationelation)
    ax.axhline(y=0,linestyle='--',color='gray')
    ax.axhline(y=-7.96/np.sqrt(max(len(x), len(y))),linestyle='--',color='gray')
    ax.axhline(y=7.96/np.sqrt(max(len(x), len(y))),linestyle='--',color='gray')
    if saveImg:
        fig.savefig('{}/crosscorrelation_{}_{}_{}.{}'.format(saveImgFolder, x.name, y.name,  saveIndex, saveImgFormat), bbox_inches='tight')

def histogram(series, colors, nBins, saveImg = False, saveIndex = 0):
    maximum = minimum = series[0].dropna().mean()
    for s in series:
        maximum = s.dropna().max() if s.dropna().max() > maximum else maximum
        minimum = s.dropna().min() if s.dropna().min() < minimum else minimum
    binCenters = np.linspace(minimum, maximum, nBins)
    fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
    for i in range(len(series)):
        ax.hist(series[i].dropna(), bins = binCenters, normed = 1, fc = colors[i], alpha=0.3, label = series[i].name)
    if len(series) == 2:
        kld = KLDiv(series[0], series[1], nBins = nBins, bins = binCenters)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')
    if len(series) == 2:
        plt.figtext(0.1,  0.010, 'KL Divergence: {}'.format(kld), size = 14, verticalalignment = 'center')

def KLDiv(p, q, nBins, bins = np.array([-1,0, 1])):
    maximum = minimum = series[0].dropna().mean()
    for s in series:
        maximum = s.dropna().max() if s.dropna().max() > maximum else maximum
        minimum = s.dropna().min() if s.dropna().min() < minimum else minimum
    [p_pdf,p_bins] = np.histogram(series[0].dropna(), bins = nBins, range = (minimum, maximum), density = True)
    [q_pdf,q_bins] = np.histogram(series[1].dropna(), bins = nBins, range = (minimum, maximum), density = True)
    kl_values = []
    for i in range(len(p_pdf)):
        if p_pdf[i] == 0 or q_pdf[i] == 0 :
            kl_values = np.append(kl_values,0)
        else:
            kl_value = np.abs(p_pdf[i]*np.log10(p_pdf[i]/q_pdf[i]))
            if np.isnan(kl_value):
                kl_values = np.append(kl_values,0)
            else:
                kl_values = np.append(kl_values,kl_value)
    return np.sum(kl_values)

# </editor-fold>

# <editor-fold> GLOBAL PARAMS
dataPath = '/home/danilofrp/projeto_final/data'
assetType = 'stocks'
asset = 'PETR4'
frequency = 'diario'

decomposeModel = 'additive'

saveImgFolder = '/home/danilofrp/projeto_final/results/preprocessing/slides'
saveImgFormat = 'png'

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

plotSeries(df['Close'], initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = '')

plotReturnSeries(df, column = 'Close', initialPlotDate='2000', finalPlotDate='2017', saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weights = None, weightModel = 'periodogram',
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weights = None, weightModel = None,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weights = None, weightModel = 'adaptative_pgram',
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, saveImg = False, saveIndex = '')

deSeason(df, column = 'Close', freq = 5, model = decomposeModel, plot = True, initialPlotDate = '2017', finalPlotDate = '2017')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 10, weights = None, weightModel = None, saveImg = False, saveIndex = '')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 10, weights = None, weightModel = 'autocorrelogram', saveImg = False, saveIndex = '')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weights = None, weightModel = 'periodogram', saveImg = False, saveIndex = '')

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weights = None, weightModel = 'adaptative_pgram', saveImg = False, saveIndex = '')

deTrendRMSE(df['2000':'2002'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weights = None, weightModel = 'autocorrelogram', saveImg = False, saveIndex = '')

deTrendRMSE(df['2000':'2002'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weights = None, weightModel = 'periodogram', saveImg = False, saveIndex = '')

deSeasonRMSE(df, column = 'Close', model = decomposeModel, maxFreq = 75, saveImg = False, saveIndex = '')

decompose(df, column = 'Close', model = decomposeModel, window = 3, freq = 5,
          plot = False, initialPlotDate = '2008', finalPlotDate = '2008')

plotPeriodogram(df['Close_EMA72_logdiff'], plotInit = 2, plotEnd = 100, yLog = False, saveImg = False, saveIndex = '')

FFT(df['Close_EMA72_logdiff'], saveImg = False, saveIndex = '')

plotSeasonalDecompose(df, column = 'Close', frequency=5, initialPlotDate='2016', finalPlotDate='2017', saveImg = False, saveIndex = '')

testStationarity(df['Close_resid'][20:], window=20, initialPlotDate='2016', finalPlotDate='2017', saveImg = False, saveIndex = '')

plotAcf(df['Close']['2000':'2002'], lags = 75, partialAcf = False, saveImg = False, saveIndex = '')

plotCrosscorrelation(df['Close/Open_returns'], df['Close_returns'], 50)

histogram([df['Close'], df['Close_trend']], colors = ['b', 'r'], nBins=100)

fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weights = None, weightModel = None, plot = False)
ax.plot(df['Close_trend'], df['Close'], 'bo')
deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weights = None, weightModel = 'periodogram', plot = False)
ax.plot(df['Close_trend'], df['Close'], 'ro', alpha=0.5)

def scatterHist(s1, s2, nBins):
    s1 = s1.dropna()
    s2 = s2.dropna()
    nullfmt = NullFormatter()

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(1, figsize=(15, 15))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.scatter(s1, s2)

    maximum = max(s1.dropna().max(), s2.dropna().max())
    minimum = min(s1.dropna().min(), s2.dropna().min())
    binCenters = np.linspace(minimum, maximum, nBins)

    axScatter.set_xlim((minimum, maximum))
    axScatter.set_ylim((minimum, maximum))

    axHistx.hist(s1, bins=nBins)
    axHisty.hist(s2, bins=nBins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

scatterHist(df['Close'], df['Close_trend'], 100)

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
#fig.savefig('{}/arima_fitted3.{}'.format(saveImgFolder, saveImgFormat), bbox_inches='tight')


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
#fig.savefig('{}/close_fitted1.{}'.format(saveImgFolder, saveImgFormat), bbox_inches='tight')

# </editor-fold>

# <editor-fold> MISC
def plotLinearFit (df, window, offset, weights = None, saveImg = False, saveIndex = ''):
    x = range(0, window)
    y = df['Close'][offset : offset + window].values
    a = np.polyfit(x, y, deg = 1, w = weights)
    fit = [.0 for i in range(window)]
    prediction = 0
    for j in range(1, -1, -1):
        prediction += a[1 - j]*(window**j)
    for i in range(window):
        fit[i] = a[1] + a[0]*x[i]
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_title('Prediction Method', fontsize = 20, fontweight = 'bold')
    ax.plot(x, y, 'bo', label="data")
    ax.plot(x, fit, 'g', label="fitted")
    ax.plot(window, prediction, 'ro', label="predicted")
    ax.plot(window, df['Close'][offset + window], 'bo')
    plt.legend()
    if saveImg:
        fig.savefig('{}/trend_fit_{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

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

window = 25
weights = list(reversed(periodogram(df['Close'].dropna())[1 : window + 1]))
plotLinearFit(df, window, 1008, weights)
