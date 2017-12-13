import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
from neuralstocks.utils import *

def plotSeries(series, title = None, ylabel = None, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    """
    Plots the desired Series

    Parameters
    ----------
    series : pandas.Series object, pandas.Series array, or pandas.DataFrame object.
        Will plot the single Series if input is a single pandas.Series, plot all
        Series if input is an array of pandas.Series, or plot all dataframe columns
        if input is a pandas.DataFrame object. All plots will overlap in the same ax

    initialPlotDate, finalPlotDate : string (yyyy or yyyy-mm) or datetime,
    indicates which period of the series to plot. If none is provided, assumes
    first and last available samples, respectively.

    saveImg : bool, indicates wheter to save the generated plot or not

    saveDir: string, directory in wich to save generated plot

    saveName : string, set of desired lenghts to calculate moving averages

    saveFormat : string, saved image format. Default 'pdf'
    """
    if isinstance(series, pd.DataFrame):
        series = [series[column] for column in series.columns.values]
    series = [series] if isinstance(series, pd.Series) else series
    initialPlotDate = series[0][initialPlotDate].index[0] if initialPlotDate else series[0].index[0]
    finalPlotDate = series[0][finalPlotDate].index[-1] if finalPlotDate else series[0].index[-1]
    if not title:
        title = '{} ({})'.format(series[0].name, initialPlotDate.strftime('%d/%m/%Y')) if initialPlotDate == finalPlotDate else '{} ({} to {})'.format(series[0].name, initialPlotDate.strftime('%d/%m/%Y'), finalPlotDate.strftime('%d/%m/%Y'))
    if not ylabel:
        ylabel = series[0].name

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1)
    fig.suptitle(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    for s in series:
        ax.plot(s[initialPlotDate:finalPlotDate])
    if saveImg:
        saveName = saveName if saveName else '{}'.format(s[0].name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotReturnSeries(df, column, asset, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    """
    Plots the desired Series, along with its return series. Return series asssumed to be present in DataFrame.

    Parameters
    ----------
    s : pandas.DataFrame object

    column: string, column to plot

    initialPlotDate, finalPlotDate : string (yyyy or yyyy-mm) or datetime,
    indicates which period of the series to plot. If none is provided, assumes
    first and last available samples, respectively.

    saveImg : bool, indicates wheter to save the generated plot or not

    saveDir: string, directory in wich to save generated plot

    saveName : string, set of desired lenghts to calculate moving averages

    saveFormat : string, saved image format. Default 'pdf'
    """
    initialPlotDate = initialPlotDate if initialPlotDate else df.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else df.index[-1]
    title = '{} {} ({})'.format(asset, column, initialPlotDate.strftime('%d/%m/%Y')) if initialPlotDate == finalPlotDate else '{} {} ({} to {})'.format(asset, column, initialPlotDate.strftime('%d/%m/%Y'), finalPlotDate.strftime('%d/%m/%Y'))
    returnName = column + '_returns'

    fig, ax = plt.subplots(figsize=(10,10), nrows = 2, ncols = 1, sharex = True)

    plt.xlabel('Date')
    ax[0].set_title(title)
    ax[0].set_ylabel('Price')
    ax[0].plot(df[column][initialPlotDate:finalPlotDate])
    ax[0].grid()

    ax[1].set_ylabel('Returns')
    ax[1].plot(df[returnName][initialPlotDate:finalPlotDate])
    ax[1].grid()

    if saveImg:
        saveName = saveName if saveName else '{}_returns'.format(column)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotLinearFit (s, window, offset = 0, weightModel = None, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    """
    Plots a demonstration of the mehtod used to extract trend

    Parameters
    ----------
    s : pandas.Series object

    window : int, number of samples of polinomial fit

    offset : int, position of the series to use as start for plot and fit. Default 0

    weightModel : string, method used to extract weights for polinomial fit.
        Accepts None, 'pgram' or 'acorr'. Default None

    saveImg : bool, indicates wheter to save the generated plot or not

    saveDir: string, directory in wich to save generated plot

    saveName : string, set of desired lenghts to calculate moving averages

    saveFormat : string, saved image format. Default 'pdf'
    """
    x = range(0, window)
    y = s[offset : offset + window].values
    if weightModel == 'pgram':
        weights = list(reversed(signal.periodogram(s[offset - 4 * window - 1 : offset].dropna())[1][1 : window + 1]))
    elif weightModel == 'acorr':
        weights = list(reversed(np.abs(autocorrelation(s[offset - window - 1: offset].dropna(), nlags= window + 1))[1 : window + 1]))
    else:
        weights = None
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
    ax.plot(window, s[offset + window], 'bo')
    plt.legend()
    if saveImg:
        saveName = saveName if saveName else 'linearFitExample'
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotDeTrendResult(df, column, window, model, weightModel, weightModelWindow, RMSE,
                      initialPlotDate = None, finalPlotDate = None, overlap = False, detailed = False,
                      saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    trendName = column + '_trend'
    residName = column + '_resid'
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
    ax[1 + int(not overlap)].set_title('Residuals')
    ax[1 + int(not overlap)].plot(df[residName][initialPlotDate:finalPlotDate])

    if detailed:
        plt.figtext(0.1,  0.010, 'deTrend Parameters:', size = 14, verticalalignment = 'center')
        plt.figtext(0.1, -0.025, 'Model: {}'.format(model), size = 14)
        plt.figtext(0.1, -0.050, 'Window size: {}'.format(window), size = 14)
        plt.figtext(0.1, -0.075, 'Weight model: {}'.format(weightModel), size = 14)
        plt.figtext(0.1, -0.100, 'Weight model window size: {}'.format(weightModelWindow), size = 14)
        plt.figtext(0.1, -0.125, 'Prediction RMSE: {}'.format(RMSE), size = 14)
    if saveImg:
        saveName = saveName if saveName else '{}_deTrend'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotPeriodogram(s, plotInit = 0, plotEnd = None, yLog = False, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    f, Pxx = signal.periodogram(s.dropna())
    plotEnd = plotEnd if plotEnd else len(s.dropna())/2

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
    fig.suptitle('Periodogram')
    ax.set_xlabel('Frequency')
    if yLog:
        plt.yscale('log')
    ax.stem(f[plotInit:plotEnd+1], Pxx[plotInit:plotEnd+1])
    if saveImg:
        saveName = saveName if saveName else '{}_periodogram'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotFFT(s, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    Fs = 1.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    y = s.dropna() - 1 if s.dropna().mean() > 0.5 else s.dropna()
    n = len(y) # lenght of the signal
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
        saveName = saveName if saveName else '{}_FFT'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotSeasonalDecompose(s, asset, frequency = 1, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    s = s.dropna()
    initialPlotDate = initialPlotDate if initialPlotDate else s.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else s.index[-1]
    title = asset + ' ' + s.name + ' (' + initialPlotDate + ')' if initialPlotDate == finalPlotDate else asset + ' ' + s.name + ' (' + initialPlotDate + ' to ' + finalPlotDate + ')'
    initialIndex = np.where(s.index == s[initialPlotDate:finalPlotDate].index[0])[0][0]
    finalIndex = np.where(s.index == s[initialPlotDate:finalPlotDate].index[-1])[0][0] + 1

    result = seasonal_decompose(s.values, model='a', freq=frequency, two_sided=False)

    fig, ax = plt.subplots(figsize=(10,15), nrows = 4, ncols = 1)

    plot_data = s[initialPlotDate:finalPlotDate]
    plt.xlabel('Date')
    ax[0].set_title(title)
    ax[0].plot(s[initialPlotDate:finalPlotDate].index,plot_data[s.name],'b-')
    #ax[0].plot(s[initialPlotDate:finalPlotDate].index,plot_data['Open'],'r:')
    #ax[0].plot(s[initialPlotDate:finalPlotDate].index,plot_data['High'],'g:')
    #ax[0].plot(s[initialPlotDate:finalPlotDate].index,plot_data['Low'],'g:')
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
        saveName = saveName if saveName else '{}_seasonalDecompose'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotAcf(s, lags = 10, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    lag_acf = autocorrelation(s, nlags=lags)

    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1)
    #Plot ACF:
    ax.set_title('Autocorrelation Function')
    ax.set_xlabel('Lags')
    ax.stem(range(1,len(lag_acf)),lag_acf[1:])
    ax.axhline(y=0,linestyle='--',color='gray')
    ax.axhline(y=-7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
    ax.axhline(y=7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')

    if saveImg:
        saveName = saveName if saveName else '{}_acf'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def testStationarity(ts, window, initialPlotDate='', finalPlotDate='', saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
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
        saveName = saveName if saveName else '{}_ADF'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotCrosscorrelation(x, y, nlags = 10, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
    """Cross correlations calculatins until nlags.
    Parameters
    ----------
    x, y : pandas.Series objects of equal lenght
    nlags : int, number of lags to calculate cross-correlation, default 10
    saveImg : bool, saves image to save directory if True, default False
    saveIndex: string, sufix to add to saved image file name, default empty
    """
    crosscorr = crosscorrelation(x, y, nlags)

    fig, ax = plt.subplots(figsize=(10,10), nrows = 1, ncols = 1)
    #Plot ACF:
    ax.set_title('Crosscorrelation ({} and {})'.format(x.name, y.name))
    ax.set_xlabel('Lags')
    ax.stem(crosscorr)
    ax.axhline(y=0,linestyle='--',color='gray')
    ax.axhline(y=-7.96/np.sqrt(max(len(x), len(y))),linestyle='--',color='gray')
    ax.axhline(y=7.96/np.sqrt(max(len(x), len(y))),linestyle='--',color='gray')
    if saveImg:
        saveName = saveName if saveName else '{}_{}_crossCorr'.format(x, y)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def histogram(series, colors, nBins, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
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
    if saveImg:
        if len(series) == 2:
            saveName = saveName if saveName else '{}_{}_hist'.format(series[1].name, series[2].name)
        else:
            saveName = saveName if saveName else '{}_hist'.format(series[1].name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def scatterHist(s1, s2, nBins, saveImg = False, saveDir = '', saveName = '', saveFormat = 'pdf'):
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

    axScatter.grid()
    axScatter.scatter(s1, s2)

    maximum = max(s1.dropna().max(), s2.dropna().max())
    minimum = min(s1.dropna().min(), s2.dropna().min())
    binCenters = np.linspace(minimum, maximum, nBins)

    slack = maximum * 0.1
    axScatter.set_xlim((minimum - slack, maximum + slack))
    axScatter.set_ylim((minimum - slack, maximum + slack))
    axScatter.plot(range(int(minimum - slack), int(maximum + slack) + 1), range(int(minimum - slack), int(maximum + slack) + 1), 'r')

    axHistx.hist(s1.dropna(), bins=binCenters, fc = 'black', label = 'Observed')
    axHisty.hist(s2.dropna(), bins=binCenters, orientation='horizontal', label = 'Predicted')

    axHistx.set_title('Observed')
    axHisty.set_title('Predicted', rotation = -90, x = 1.05, y = 0.5)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    if saveImg:
        saveName = saveName if saveName else '{}_{}_scatterHist'.format(s1, s2)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')
