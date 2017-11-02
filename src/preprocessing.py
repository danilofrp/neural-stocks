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
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/returns{}.png'.format(asset, saveIndex), bbox_inches='tight')

def plot_periodogram(df, column, initialLag = 0, numberOfLags = 30, yLog = False, saveImg = False, saveIndex = ''):
    if isnan(df[column].iloc[0]):
        df = df.drop(df.index[0])
    pgram = periodogram(df[column])
    length = len(pgram) if len(pgram) < numberOfLags else numberOfLags + 1

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
    plot_data = pgram[initialLag:length]
    plt.xlabel('Lags')
    ax.set_title('Periodogram')
    if yLog:
        plt.yscale('log')
    ax.stem(range(initialLag, length), plot_data)

    if saveImg:
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/periodogram{}.png'.format(asset, saveIndex), bbox_inches='tight')

def plot_seasonalDecompose(df, asset, column, initialPlotDate = '', finalPlotDate = '', frequency = 1, saveImg = False, saveIndex = ''):
    if isnan(df[column].iloc[0]):
        df = df.drop(df.index[0])
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    title = asset + ' ' + column + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' ' + column + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'
    initialIndex = np.where(df.index == df[initialPlotDate:finalPlotDate].index[0])[0][0]
    finalIndex = np.where(df.index == df[initialPlotDate:finalPlotDate].index[-1])[0][0] + 1

    result = seasonal_decompose(df[column].values, model='additive', freq=frequency, two_sided=False)

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
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/seasonal_decompose{}.png'.format(asset, saveIndex), bbox_inches='tight')

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
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/test_stationarity{}.png'.format(asset, saveIndex), bbox_inches='tight')

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
        fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/acf_pacf{}.png'.format(asset, saveIndex), bbox_inches='tight')
# </editor-fold>

# <editor-fold> DATA INFO
dataPath = '/home/danilofrp/projeto_final/data'
assetType = 'forex'
asset = 'USDBRL'
frequency = 'diario'
# </editor-fold>

df = acquireData(dataPath, assetType, asset, frequency, replicateForHolidays = True)

plot_returnSeries(df, asset, initialPlotDate='2016', finalPlotDate='2016', saveImg = True, saveIndex = '')

plot_periodogram(df, 'Close', initialLag = 0, numberOfLags = 30, yLog = False, saveImg = True, saveIndex = '1')

plot_seasonalDecompose(df, asset, 'Close_r', initialPlotDate='2016', finalPlotDate='2016', frequency=10, saveImg = True, saveIndex = '2')

test_stationarity(df['Close_r'][:'2016'], window=10, initialPlotDate='2016', finalPlotDate='2016', saveImg = True, saveIndex = '1')

plot_acfAndPacf(df['Close_r'], 10, saveImg = True, saveIndex = '1')

model = ARIMA(df['Close_r'][:'2016'], order=(2, 1, 1))
results_ARIMA = model.fit(disp=-1)
fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
#ax.plot(results_ARIMA.resid['2016-07':'2016-12'])
ax.plot(df['Close_r']['2016-07':'2016-12'])
ax.plot(-results_ARIMA.fittedvalues['2016-07':'2016-12'].shift(-1), color='red')
ax.axhline(y=0,linestyle='--',color='gray')
ax.set_title('RSS: %.4f'% sum((-results_ARIMA.fittedvalues['2016-07':'2016-12'].shift(-1)-df['Close_r']['2016-07':'2016-12'])**2))
fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/arima_fitted4.png'.format(asset), bbox_inches='tight')


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
fig.savefig('/home/danilofrp/projeto_final/results/preprocessing/{}/close_fitted1.png'.format(asset), bbox_inches='tight')
