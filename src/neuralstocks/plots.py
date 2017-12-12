import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from statsmodels.tsa import stattools
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
from neuralstocks.utils import *

def plotSeries(s, asset, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    """
    Plots the desired Series

    Parameters
    ----------
    s : pandas.Series object

    initialPlotDate, finalPlotDate : string (yyyy or yyyy-mm) or datetime,
    indicates which period of the series to plot. If none is provided, assumes
    first and last available samples, respectively.

    saveImg : bool, indicates wheter to save the generated plot or not

    saveDir: string, directory in wich to save generated plot

    saveName : string, set of desired lenghts to calculate moving averages

    saveFormat : string, saved image format. Default '.pdf'
    """
    initialPlotDate = initialPlotDate if initialPlotDate else s.index[0]
    finalPlotDate = finalPlotDate if finalPlotDate else s.index[-1]
    title = '{} {} ({})'.format(asset, s.name, initialPlotDate.strftime('%d/%m/%Y')) if initialPlotDate == finalPlotDate else '{} {} ({} to {})'.format(asset, s.name, initialPlotDate.strftime('%d/%m/%Y'), finalPlotDate.strftime('%d/%m/%Y'))

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1)
    fig.suptitle(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.plot(s[initialPlotDate:finalPlotDate])
    if saveImg:
        saveName = saveName if saveName else '{}'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotReturnSeries(df, column, asset, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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

    saveFormat : string, saved image format. Default '.pdf'
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

def plotLinearFit (s, window, offset = 0, weightModel = None, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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

    saveFormat : string, saved image format. Default '.pdf'
    """
    x = range(0, window)
    y = s[offset : offset + window].values
    if weightModel == 'pgram':
        weights = list(reversed(signal.periodogram(s[offset - 4 * window - 1 : offset].dropna())[1][1 : window + 1]))
    elif weightModel == 'acorr':
        weights = list(reversed(np.abs(stattools.acf(s[offset - window - 1: offset].dropna(), nlags= window + 1))[1 : window + 1]))
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

def plotPeriodogramStats(s, plotInit = 0, plotEnd = None, yLog = False, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    pgram = stattools.periodogram(s.dropna())
    plotEnd = plotEnd if plotEnd else len(s.dropna())/2

    fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1, sharex = True)
    fig.suptitle('Periodogram')
    # ax.set_xlabel('Period (samples)')
    if yLog:
        plt.yscale('log')
    ax.stem(range(plotInit,plotEnd+1), pgram[plotInit:plotEnd+1])
    if saveImg:
        saveName = saveName if saveName else '{}_periodogram'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotPeriodogramSciPy(s, plotInit = 0, plotEnd = None, yLog = False, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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

def plotFFT(s, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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
        saveName = saveName if saveName else '{}_FFT'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotSeasonalDecompose(s, asset, frequency = 1, initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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

def plotAcf(s, lags = 10, partialAcf = False, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    lag_acf = stattools.acf(s.dropna(), nlags=lags)
    if partialAcf:
        lag_pacf = stattools.pacf(s.dropna(), nlags=lags, method='ols')

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
        ax[1].stem(range(1,len(lag_pacf)),lag_pacf[1:])
        ax[1].axhline(y=0,linestyle='--',color='gray')
        ax[1].axhline(y=-7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')
        ax[1].axhline(y=7.96/np.sqrt(len(s.dropna())),linestyle='--',color='gray')

    if saveImg:
        saveName = saveName if saveName else '{}_acf'.format(s.name)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')

def plotCrosscorrelation(x, y, nlags = 10, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
    """Cross correlations calculatins until nlags.
    Parameters
    ----------
    x, y : pandas.Series objects of equal length
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

def histogram(series, colors, nBins, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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

def scatterHist(s1, s2, nBins, saveImg = False, saveDir = '', saveName = '', saveFormat = '.pdf'):
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

    axScatter.set_xlim((minimum - 1, maximum + 1))
    axScatter.set_ylim((minimum - 1, maximum + 1))

    axHistx.hist(s1.dropna(), bins=binCenters, fc = 'black', label = 'Observed')
    axHisty.hist(s2.dropna(), bins=binCenters, orientation='horizontal', label = 'Predicted')

    axHistx.set_title('Observed')
    axHisty.set_title('Predicted', rotation = -90, x = 1.05, y = 0.5)
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    if saveImg:
        saveName = saveName if saveName else '{}_{}_scatterHist'.format(s1, s2)
        fig.savefig('{}/{}.{}'.format(saveDir, saveName, saveFormat), bbox_inches='tight')
