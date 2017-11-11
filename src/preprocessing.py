# <editor-fold> IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import periodogram, adfuller, acf, pacf
from math import isnan
%matplotlib inline
# </editor-fold>

# <editor-fold> FUNCTIONS DEF
def insertMissingDays(df):
    start = datetime(2000, 3, 1)
    end = datetime(2017, 9, 1)
    step = timedelta(days=1)

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

def acquireData(path, assetType, asset, samplingFrequency, replicateForHolidays = False):
    filepath = path + '/' + assetType + '/' + asset + '/' + samplingFrequency + '/' + asset + '.CSV'

    df = pd.read_csv(filepath, delimiter=';', decimal=',',
                     parse_dates=['Date'], dayfirst=True, index_col='Date')
    df = df.sort_index() #csv entries begin from most recent to older dates

    if replicateForHolidays:
        df.loc[:,'Holiday'] = 0
        df = insertMissingDays(df)

    df['Close_r'] = np.log(df.Close/df.Close.shift(1))

    return df.drop(df.index[0])

def plot_returnSeries(df, asset, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = ''):
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    title = asset + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'

    fig, ax = plt.subplots(figsize=(10,10), nrows = 2, ncols = 1, sharex = True)

    plot_data = df[initialPlotDate:finalPlotDate]
    plt.xlabel('Date')
    ax[0].set_title(title)
    ax[0].set_ylabel('Price')
    ax[0].plot(plot_data['Close'])
    ax[0].grid()

    ax[1].set_ylabel('Returns')
    ax[1].plot(plot_data['Close_r'])
    ax[1].grid()

    if saveImg:
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/returns{}.pdf'.format(asset, saveIndex), bbox_inches='tight')

def deTrend(df, column, window, fitOrder = 1, plot = False, initialPlotDate = None, finalPlotDate = None):
    if window < fitOrder + 1:
        window = fitOrder +1
        print 'Warning: window must be at least {} samples wide for a fit of order {}. Adjusting window for minimal value.'.format(fitOrder+1, fitOrder)
    trendName = 'trend_' + column
    residualName = 'residual_' + column
    df[trendName] = np.empty(len(df[column]))*np.nan
    x = range(0, window)
    for i in range(0, len(df[column])):
        if i < (window):
            df[trendName].iloc[i] = np.nan
        else:
            y = df[column][(i - window):i].values
            a = np.polyfit(x, y, fitOrder)
            prediction = 0
            for j in range(fitOrder, -1, -1):
                prediction += a[fitOrder - j]*(window**j)
            df.set_value(df.index[i], trendName, prediction)
    df[residualName] = df[column] / df[trendName]

    if plot:
        initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
        finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
        fig, ax = plt.subplots(figsize=(15,10), nrows = 2, ncols = 1, sharex = True)
        plt.xlabel('Date')
        ax[0].set_title('Trend Predictions')
        ax[0].set_ylabel('Price')
        ax[0].plot(df[column][initialPlotDate:finalPlotDate])
        ax[0].plot(df[trendName][initialPlotDate:finalPlotDate])
        ax[1].plot(df[residualName][initialPlotDate:finalPlotDate])

def plot_deTrend_RSS(df, column, fitOrder = 1, windowMaxSize = 30, saveImg = False, saveIndex = ''):
    df2 = df.copy()
    RSS = np.empty(windowMaxSize + 1)*np.nan
    for i in range(fitOrder + 1, windowMaxSize + 1):
        deTrend(df2, column = column, window = i, fitOrder = fitOrder)
        RSS[i] = np.square(df2['residual_{}'.format(column)]).sum()
    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1, sharex = True)
    ax.set_title('RSS for each detrend window size')
    ax.set_xlabel('Window size')
    ax.set_ylabel('RSS')
    ax.plot(range(0,windowMaxSize+1), RSS, 'bo')
    minValue = min(RSS[fitOrder + 1 : windowMaxSize + 1])
    for i in range(fitOrder + 1, windowMaxSize + 1):
        if RSS[i] == minValue:
            minIndex = i
    plt.annotate('local min', size = 18, xy=(minIndex*1.01, minValue*1.01), xytext=(minIndex*1.1, minValue*1.1), arrowprops=dict(facecolor='black', shrink=0.05))
    if saveImg:
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/detrend_RSS{}.pdf'.format(asset, saveIndex), bbox_inches='tight')

def plot_periodogram(df, column, numberOfLags = 30, initialLag = 0, yLog = False, saveImg = False, saveIndex = ''):
    if isnan(df[column].iloc[0]):
        df = df.drop(df.index[0])
    pgram = periodogram(df[column])
    length = len(pgram) if len(pgram) < numberOfLags else numberOfLags + 1

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
    plt.xlabel('Lags')
    ax.set_title('Periodogram')
    if yLog:
        plt.yscale('log')
    ax.stem(range(initialLag, length), pgram[initialLag:length])

    if saveImg:
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/periodogram{}.pdf'.format(asset, saveIndex), bbox_inches='tight')

def plot_seasonalDecompose(df, asset, column, initialPlotDate = '', finalPlotDate = '', frequency = 1, saveImg = False, saveIndex = ''):
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
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/seasonal_decompose{}.pdf'.format(asset, saveIndex), bbox_inches='tight')

def test_stationarity(ts, window, initialPlotDate='', finalPlotDate='', saveImg = False, saveIndex = ''):
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
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/test_stationarity{}.pdf'.format(asset, saveIndex), bbox_inches='tight')

def plot_acfAndPacf(df, lags = 10, saveImg = False, saveIndex = ''):
    lag_acf = acf(df, nlags=lags)
    lag_pacf = pacf(df, nlags=lags, method='ols')

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 2)
    print 'acf: {}, {}, {}, {}, {}, {}'.format(lag_acf[0], lag_acf[1], lag_acf[2], lag_acf[3], lag_acf[4], lag_acf[5], )
    print 'pacf: {}, {}, {}, {}, {}, {}'.format(lag_pacf[0], lag_pacf[1], lag_pacf[2], lag_pacf[3], lag_pacf[4], lag_pacf[5], )

    #Plot ACF:
    ax[0].stem(lag_acf)
    ax[0].axhline(y=0,linestyle='--',color='gray')
    ax[0].axhline(y=-7.96/np.sqrt(len(df)),linestyle='--',color='gray')
    ax[0].axhline(y=7.96/np.sqrt(len(df)),linestyle='--',color='gray')
    ax[0].set_title('Autocorrelation Function')

    #Plot PACF:
    ax[1].stem(lag_pacf)
    ax[1].axhline(y=0,linestyle='--',color='gray')
    ax[1].axhline(y=-7.96/np.sqrt(len(df)),linestyle='--',color='gray')
    ax[1].axhline(y=7.96/np.sqrt(len(df)),linestyle='--',color='gray')
    ax[1].set_title('Partial Autocorrelation Function')
    plt.tight_layout()

    if saveImg:
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/acf_pacf{}.pdf'.format(asset, saveIndex), bbox_inches='tight')
# </editor-fold>

# <editor-fold> DATA INFO
dataPath = '/home/danilofrp/projeto_final/data'
assetType = 'stocks'
asset = 'PETR4'
frequency = 'diario'
# </editor-fold>

df = acquireData(dataPath, assetType, asset, frequency, replicateForHolidays = True)

plot_returnSeries(df, asset, initialPlotDate='2016-10', finalPlotDate='2016-12', saveImg = False, saveIndex = '1')

deTrend(df, column = 'Close', window = 8, fitOrder = 1, plot = True, initialPlotDate = '2000', finalPlotDate = '2017')

plot_periodogram(df, 'Close', numberOfLags = 30, initialLag = 0, yLog = False, saveImg = False, saveIndex = '4')

plot_seasonalDecompose(df, asset, 'Close', initialPlotDate='2016', finalPlotDate='2017', frequency=5, saveImg = False, saveIndex = '5')

test_stationarity(df['Close_r'][:'2016'], window=10, initialPlotDate='2016', finalPlotDate='2016', saveImg = False, saveIndex = '1')

plot_acfAndPacf(df['Close_r'], 10, saveImg = False, saveIndex = '1')

# <editor-fold> ARIMA
model = ARIMA(df['Close_r'][:'2016'], order=(2, 1, 1))
results_ARIMA = model.fit(disp=-1)
fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
#ax.plot(results_ARIMA.resid['2016-07':'2016-12'])
ax.plot(df['Close_r']['2016-07':'2016-12'])
ax.plot(results_ARIMA.fittedvalues['2016-07':'2016-12'], color='red')
ax.axhline(y=0,linestyle='--',color='gray')
ax.set_title('RSS: %.4f'% sum((results_ARIMA.fittedvalues['2016-07':'2016-12']-df['Close_r']['2016-07':'2016-12'])**2))
#fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/arima_fitted3.pdf'.format(asset), bbox_inches='tight')


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
#fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/close_fitted1.pdf'.format(asset), bbox_inches='tight')

# </editor-fold>

# <editor-fold> MISC

window = 4
offset = 0
data = df['Close'][0+offset:window+1+offset].values
a = np.polyfit(range(0, window), data[0:window], 1)
fit = np.empty(window)*np.nan
for i in range(0, window):
    fit[i] = a[0]*i + a[1]
prediction = a[0]*window + a[1]
fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
ax.scatter(range(0,window+1), data)
ax.plot(range(0, window), fit, color='r')
ax.plot(window, prediction, 'go')

print df[['Close', 'trend_Close']]

plot_deTrend_RSS(df, 'Close', fitOrder = 1, windowMaxSize = 15)

from pandas.core.nanops import nanmean as pd_nanmean

freq = 5
seasonal = pd.Series([np.nan], [df['Close'].index])
for i in range(freq):
    for j in range(i, len(df['residual_Close']), freq):
        seasonal[i] = pd_nanmean(df['residual_Close'][i-freq::freq])
#aux = np.array([pd_nanmean(df['residual_Close'][i::freq]) for i in range(freq)])
#seasonal = np.tile(aux, len(df['Close']) // freq + 1)[4:len(df['Close'])]
fig, ax = plt.subplots(figsize=(10,10), nrows = 2, ncols = 1, sharex = True)
ax[0].plot(df['residual_Close'][:200].index, df['residual_Close'][:200])
ax[1].plot(df['residual_Close'][:200].index, seasonal[:200])
# </editor-fold>


np.isnan(df['residual_Close'][2])
print seasonal

print df['resitual_close']
print pd_nanmean(df['residual_Close'][5-freq::5])

range(0, 10, 5)
