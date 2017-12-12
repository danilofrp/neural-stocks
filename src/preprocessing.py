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
from neuralstocks.preprocessing import *
from neuralstocks.plots import *
from neuralstocks.utils import *
from __future__ import print_function
%matplotlib inline
# </editor-fold>

# <editor-fold> FUNCTIONS DEF

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
            plot = False, initialPlotDate = None, finalPlotDate = None, overlap = False, detailed = False, saveImg = False, saveIndex = ''):
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
            if weightModel.startswith('window_'):
                weights, weightModelWindow = getWeights(df[column][i - weightModelWindow - 1 : i], window = window, weightModel = weightModel, weightModelWindow = weightModelWindow)
            df.set_value(df.index[i], trendName, predict(x = range(0, window), y = df[column][(i - window):i].values, fitOrder = fitOrder, weights = weights, window = window))

    df[residName] = df[column] / df[trendName] if model.startswith('m') else df[column] - df[trendName]
    RMSE = (np.square(df['{}_resid'.format(column)].dropna() - int(model.startswith('m'))).sum())/(len(df.dropna()))

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

        if detailed:
            plt.figtext(0.1,  0.010, 'deTrend Parameters', size = 14, verticalalignment = 'center')
            plt.figtext(0.1, -0.025, 'Model: {}'.format(model), size = 14)
            plt.figtext(0.1, -0.050, 'Window size: {}'.format(window), size = 14)
            plt.figtext(0.1, -0.075, 'Weight model: {}'.format(weightModel), size = 14)
            plt.figtext(0.1, -0.100, 'Weight model window size: {}'.format(weightModelWindow), size = 14)
            plt.figtext(0.1, -0.125, 'Prediction RMSE: {}'.format(RMSE), size = 14)
        if saveImg:
            fig.savefig('{}/deTrend_result{}.{}'.format(saveDir, saveIndex, saveFormat), bbox_inches='tight')

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

def deTrendRMSE(df, column, model = 'additive', fitOrder = 1, windowMaxSize = 30, weightModel = None, weightModelWindow = None, saveImg = False, saveIndex = ''):
    df2 = df.copy()
    model = 'multiplicative' if model.startswith('m') else 'additive'
    RMSE = np.empty(windowMaxSize + 1)*np.nan
    for i in range(fitOrder + 1, windowMaxSize + 1):
        deTrend(df2, column = column, window = i, model = model, fitOrder = fitOrder, weightModel = weightModel, weightModelWindow = weightModelWindow)
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

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = None, weightModelWindow = 25,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, detailed = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weightModel = 'full_pgram', weightModelWindow = 25,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, detailed = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = 'window_pgram', weightModelWindow = 100,
            plot = True, initialPlotDate = '2008-07', finalPlotDate = '2008-12', overlap = True, detailed = True, saveImg = False, saveIndex = '')

deTrend(df, column = 'Close', window = 10, model = decomposeModel, fitOrder = 1, weightModel = 'window_acorr', weightModelWindow = 25,
            plot = True, initialPlotDate = '2008-07', finalPlotDate = '2008-12', overlap = True, detailed = True, saveImg = False, saveIndex = '')

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
deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = 'full_pgram', weightModelWindow = None)
ax.plot(df['Close_trend'], df['Close'], 'bo')
deTrend(df, column = 'Close', window = 10, model = decomposeModel, fitOrder = 1, weightModel = 'window_pgram', weightModelWindow = 25)
ax.plot(df['Close_trend'], df['Close'], 'ro', alpha=0.5)

scatterHist(df['Close'], df['Close_trend'], nBins = 100, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

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
