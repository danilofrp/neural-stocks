# <editor-fold> IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from pandas.core.nanops import nanmean as pd_nanmean
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

def acquireData(replicateForHolidays = False):
    filepath = dataPath + '/' + assetType + '/' + asset + '/' + frequency + '/' + asset + '.CSV'

    df = pd.read_csv(filepath, delimiter=';', decimal=',',
                     parse_dates=['Date'], dayfirst=True, index_col='Date')
    df = df.sort_index() #csv entries begin from most recent to older dates

    if replicateForHolidays:
        df.loc[:,'Holiday'] = 0
        df = insertMissingDays(df)

    df['Close_r'] = np.log(df.Close/df.Close.shift(1))

    return df.drop(df.index[0])

def plot_Series(df, column, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = ''):
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0].strftime('%d-%m-%Y')
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1].strftime('%d-%m-%Y')
    title = '{} {} ({})'.format(asset, column, initialPlotDate) if initialPlotDate == finalPlotDate else '{} {} ({} to {})'.format(asset, column, initialPlotDate, finalPlotDate)

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1)
    fig.suptitle(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.plot(df[column][initialPlotDate:finalPlotDate])
    if saveImg:
        fig.savefig('{}/{}{}.{}'.format(saveImgFolder, column, saveIndex, saveImgFormat), bbox_inches='tight')

def plot_returnSeries(df, column, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = ''):
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    title = asset + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'
    returnName = column + '_r'

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

def deTrend(df, column, window, model = 'additive', fitOrder = 1, plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    if window < fitOrder + 1:
        window = fitOrder +1
        print 'Warning: window must be at least {} samples wide for a fit of order {}. Adjusting window for minimal value.'.format(fitOrder+1, fitOrder)
    trendName = column + '_trend'
    residName = column + '_resid'
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
    if model == 'multiplicative':
        df[residName] = df[column] / df[trendName]
    else :
        df[residName] = df[column] - df[trendName]

    if plot:
        initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
        finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
        fig, ax = plt.subplots(figsize=(15,10), nrows = 3, ncols = 1, sharex = True)
        plt.xlabel('Date')
        ax[0].set_title('Observed')
        ax[0].plot(df[column][initialPlotDate:finalPlotDate])
        ax[1].set_title('Trend Predictions (n = {}, {} model)'.format(window, model))
        ax[1].plot(df[trendName][initialPlotDate:finalPlotDate])
        ax[2].set_title('Residuals ({} model)'.format(model))
        ax[2].plot(df[residName][initialPlotDate:finalPlotDate])
        if saveImg:
            fig.savefig('{}/deTrend_result{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def deSeason(df, column, freq, model = 'additive', plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveIndex = ''):
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
        seasonal_means = seasonal_mean(df, residName, freq)
        for i in range(len(df[seasonalName])):
            df.set_value(df.index[i], seasonalName, seasonal_means[i%freq])
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

def seasonal_mean(df, column, freq):
    return np.array([pd_nanmean(df[column][i::freq]) for i in range(freq)])

def decompose(df, column, model = 'additive', window = 3, fitOrder = 1, freq = 5, plot = False, initialPlotDate = None, finalPlotDate = None, saveImg = False, saveIndex = ''):
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

def plot_deTrend_RSS(df, column, model = 'additive', fitOrder = 1, windowMaxSize = 30, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    df2 = df.copy()
    RSS = np.empty(windowMaxSize + 1)*np.nan
    for i in range(fitOrder + 1, windowMaxSize + 1):
        deTrend(df2, column = column, window = i, model = model, fitOrder = fitOrder)
        if model == 'multiplicative':
            RSS[i] = np.square((df2['{}_resid'.format(column)] - 1)).sum() # subtrair a média
        else:
            RSS[i] = np.square(df2['{}_resid'.format(column)]).sum()
    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1, sharex = True)
    ax.set_title('RSS for each deTrend window size ({} model)'.format(model))
    ax.set_xlabel('Window size')
    ax.set_ylabel('RSS')
    ax.plot(range(0,windowMaxSize+1), RSS, 'bo')
    minValue = min(RSS[fitOrder + 1 : windowMaxSize + 1])
    for i in range(fitOrder + 1, windowMaxSize + 1):
        if RSS[i] == minValue:
            minIndex = i
    plt.annotate('local min', size = 18, xy=(minIndex, minValue), xytext=(minIndex*1.1, minValue*1.1), arrowprops=dict(facecolor='black', shrink=0.05))
    if saveImg:
        fig.savefig('{}/deTrend_RSS{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def plot_deSeason_RSS(df, column, model ='additive', maxFreq = 20, saveImg = False, saveIndex = ''):
    model = 'multiplicative' if model.startswith('m') else 'additive'
    df2 = df.copy()
    RSS = np.empty(maxFreq + 1)*np.nan
    for i in range(0, maxFreq + 1):
        deSeason(df2, column = column, freq = i, model = model)
        if model == 'multiplicative':
            RSS[i] = np.square((df2['{}_resid'.format(column)] - 1)).sum()
        else:
            RSS[i] = np.square(df2['{}_resid'.format(column)]).sum()
    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1, sharex = True)
    ax.set_title('RSS for each deSeason frequency ({} model)'.format(model))
    ax.set_xlabel('Frequency (days)')
    ax.set_ylabel('RSS')

    ax.plot(range(0,maxFreq+1), RSS, 'bo')
    minValue = min(RSS[0 : maxFreq + 1])
    for i in range(0, maxFreq + 1):
        if RSS[i] == minValue:
            minIndex = i
    plt.annotate('local min', size = 18, xy=(minIndex, minValue), xytext=(minIndex, minValue), arrowprops=dict(facecolor='black', shrink=0.05))
    if saveImg:
        fig.savefig('{}/deSeason_RSS{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def plot_periodogram(df, column, plotInit = 0, plotEnd = None, yLog = False, saveImg = False, saveIndex = ''):
    if isnan(df[column].iloc[0]):
        df = df.drop(df.index[0])
    pgram = periodogram(df[column])
    plotEnd = plotEnd if plotEnd else len(df)/2
    Fs = 1.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    n = len(df) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)]

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
    ax.set_xlabel('Frequency (1/sample)')
    ax.set_title('Periodogram')
    if yLog:
        plt.yscale('log')
    ax.stem(frq[plotInit:plotEnd], pgram[plotInit:plotEnd])

    if saveImg:
        fig.savefig('{}/periodogram{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

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
        fig.savefig('{}/seasonal_decompose{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

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
        fig.savefig('{}/test_stationarity{}.{}'.format(saveImgFolder, saveIndex, saveImgFormat), bbox_inches='tight')

def plot_acfAndPacf(df, lags = 10, saveImg = False, saveIndex = ''):
    lag_acf = acf(df, nlags=lags)
    lag_pacf = pacf(df, nlags=lags, method='ols')

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 2)
    #print 'acf: {}, {}, {}, {}, {}, {}'.format(lag_acf[0], lag_acf[1], lag_acf[2], lag_acf[3], lag_acf[4], lag_acf[5], )
    #print 'pacf: {}, {}, {}, {}, {}, {}'.format(lag_pacf[0], lag_pacf[1], lag_pacf[2], lag_pacf[3], lag_pacf[4], lag_pacf[5], )

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
        fig.savefig('{}/acf_pacf{}.{}'.format(asset, saveIndex, saveImgFormat), bbox_inches='tight')
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
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
# </editor-fold>

# <editor-fold> workspace
df = acquireData(replicateForHolidays = True)

plot_Series(df, column = 'Close', initialPlotDate = '', finalPlotDate = '', saveImg = False, saveIndex = '')

plot_returnSeries(df, initialPlotDate='2017-05', finalPlotDate='2017-05', saveImg = False, saveIndex = '1')

deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, plot = True, initialPlotDate = '2000', finalPlotDate = '2017', saveImg = False, saveIndex = '10')

deSeason(df, column = 'Close', freq = 5, model = decomposeModel, plot = True, initialPlotDate = '2017', finalPlotDate = '2017')

plot_deTrend_RSS(df, column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 15, saveImg = False, saveIndex = '')

plot_deSeason_RSS(df, column = 'Close', model = decomposeModel, maxFreq = 300, saveImg = False, saveIndex = '')

decompose(df, column = 'Close', model = decomposeModel, window = 3, freq = 5, plot = True, initialPlotDate = '2008', finalPlotDate = '2008')

plot_periodogram(df[4:], column = 'Close_resid', plotInit = 0, plotEnd = '', yLog = False, saveImg = False, saveIndex = '10')

plot_seasonalDecompose(df, column = 'Close', initialPlotDate='2016', finalPlotDate='2017', frequency=5, saveImg = False, saveIndex = '5')

test_stationarity(df['Close_resid'][20:], window=20, initialPlotDate='2016', finalPlotDate='2017', saveImg = False, saveIndex = '1')

plot_acfAndPacf(df['Close_resid'][20:], lags = 60, saveImg = False, saveIndex = '10')

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

offset = 101
window = 4
x = range(0, window)
y = df['Close'][offset : offset + window].values
a = np.polyfit(x, y, 1)
fit = [0.0 for i in range(window)]
prediction = 0
for j in range(1, -1, -1):
    prediction += a[1 - j]*(window**j)
for i in range(window):
    fit[i] = a[1] + a[0]*x[i]
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.plot(x, y, 'bo', label="data")
ax.plot(x, fit, 'g', label="fitted")
ax.plot(window, prediction, 'ro', label="predicted")
ax.plot(window, df['Close'][offset + window], 'bo')
plt.legend()
fig.savefig('{}/trend_fit.{}'.format(saveImgFolder, saveImgFormat), bbox_inches='tight')

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

print df['Close_resid'][4:].head()

Fs = 1.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
y = df['Close_resid'][4:]

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
ax[1].set_xlabel('Freq (1/day)')
ax[1].set_ylabel('|X(freq)|')
fig.savefig('{}fft_resid.{}'.format(saveImgFolder, saveImgFormat), bbox_inches='tight')
