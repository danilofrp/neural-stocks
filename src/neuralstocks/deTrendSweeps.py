# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from neuralstocks.dataacquisition import *
from neuralstocks.utils import *
from neuralstocks.preprocessing import *

def deTrendAcorrGreedySearch(asset, dataPath, savePath, windowMinSize = 3, weightModelWindowMinSize = 6, windowMaxSize = 25, weightModelWindowMaxSize = 150, bot = None, verbose = False, dev = False):
    saveVarPath = savePath + '/Variables'
    saveFigPath = savePath + '/Figures'
    if dev: windowMaxSize = 5
    if dev: weightModelWindowMaxSize = 100
    init_time = time.time()
    filePath = dataPath + '/stocks/{}/diario/{}.CSV'.format(asset, asset)
    df = acquireData(filePath = filePath, replicateForHolidays = True, force = True)
    if dev: df = df[-(2*windowMaxSize + weightModelWindowMaxSize):]
    end_time = time.time()
    if verbose: print ('Time to acquire data: '+str(end_time-init_time)+' seconds')

    init_time = time.time()
    df2 = df.copy()
    saveName = '{}_RMSEanalysis_acf_w{}m{}'.format(asset, windowMaxSize, weightModelWindowMaxSize)
    if dev: saveName = saveName + '_dev'
    model = 'additive'
    column = 'Close'
    minimal = np.inf

    RMSE = np.empty((windowMaxSize + 1, weightModelWindowMaxSize + 1), dtype=float)*np.nan
    for i in range(windowMinSize, windowMaxSize + 1):
        for j in range(2 * i, weightModelWindowMaxSize + 1):
            i_time = time.time()
            deTrend(df2, column = column, window = i, model = model, fitOrder = 1, weightModel = 'window_acorr', weightModelWindow = j)
            e_time = time.time()
            if verbose: print('Completed {} deTrend ({}, {}). Time to run deTrend: {} seconds'.format(asset, i, j, str(e_time-i_time)), end='\r')
            if model.startswith('m'):
                RMSE[i, j] = np.sqrt(np.square((df2['{}_resid'.format(column)] - 1)).sum()/len(df2['{}_resid'.format(column)].dropna()))
            else:
                RMSE[i, j] = np.sqrt(np.square(df2['{}_resid'.format(column)]).sum()/len(df2['{}_resid'.format(column)].dropna()))
            if RMSE[i, j] < minimal:
                minimal = RMSE[i][j]
                iMin = i
                jMin = j

    if verbose: print('Minimal {} RSME = {}, at window = {} and weightModelWindow = {} * window'.format(asset, minimal, iMin, jMin))
    RMSE_backup = RMSE
    try:
        joblib.dump(RMSE, '{}/{}.pkl'.format(saveVarPath, saveName))
        if verbose: print('Dump succesfull!')
    except:
        print('[!] Warning: Dump failed for {}!'.format(asset))

    end_time = time.time()
    if verbose: print('Time to run {} {}x{} acorr sweep deTrend analysis: {} seconds'.format(asset, windowMaxSize, weightModelWindowMaxSize, str(end_time-init_time)))

    fig, ax = plt.subplots(figsize=(10,7.5))
    cax = ax.imshow(RMSE[:,:], cmap="jet", aspect="auto")
    plt.gca().invert_yaxis()
    ax.set_title('RMSE analysis: \nAutocorrelation x Trend greedy search')
    ax.set_ylim([windowMinSize + 0.5, windowMaxSize + 0.5])
    ax.set_xlim([weightModelWindowMinSize + 0.5, weightModelWindowMaxSize + 0.5])
    ax.set_xlabel('Past samples used for autocorrelation estimation')
    ax.set_ylabel('Past samples used for trend estimation')
    cbar = fig.colorbar(cax)
    plt.figtext(0.45,  -0.045, 'Minimal RMSE: {:.5f}, using Trend window of {} samples \nand autocorrelation window of {} samples'.format(minimal, iMin, jMin), size = 16, horizontalalignment = 'center')

    fig.savefig('{}/{}{}.pdf'.format(saveFigPath, saveName, '_dev' if dev else None), bbox_inches='tight')
    fig.savefig('{}/{}{}.png'.format(saveFigPath, saveName, '_dev' if dev else None), bbox_inches='tight')

    if bot:
        message  = 'Autocorrelation deTrend greedy search ({}, {}x{}) completed'.format(asset, windowMaxSize, weightModelWindowMaxSize)
        imgPath  = '{}/{}{}.png'.format(saveFigPath, saveName, '_dev' if dev else None)
        filePath = '{}/{}{}.pdf'.format(saveFigPath, saveName, '_dev' if dev else None)
        bot.sendMessage(message, imgPath, filePath)

    return {'asset': asset, 'optimalTrendSamples': iMin, 'optimalAcorrSamples': jMin, 'optimalRMSE': minimal}

def deTrendOptimal(asset, dataPath, savePath, bot = None):
    filePath = dataPath + '/stocks/' + asset + '/diario/' + asset + '.CSV'
    deTrendParams = joblib.load('{}/allAcorrSweepResults.pkl'.format(savePath + '/Variables'))[asset]
    savePath = savePath.split('ns-results', 1)[0] + 'ns-results/data/preprocessed/diario'
    # if not os.path.exists(savePath): os.makedirs(savePath)
    try:
        os.makedirs(savePath)
        break
    except OSError, e:
        if e.errno != os.errno.EEXIST:
            raise
        # time.sleep might help here
        pass

    df = acquireData(filePath = filePath,
                     replicateForHolidays = True,
                     meanStdLen = 20,
                     returnCalcParams = [['Close'], ['Close', 'Open'], ['High', 'Close' ], ['Low', 'Close']],
                     EMAparams = [{'column': 'Close', 'lenght': 17},
                                  {'column': 'Close', 'lenght': 72},
                                  {'column': 'Close', 'lenght': 200},
                                  {'column': 'Volume', 'lenght': 21}],
                     MACDParams = [{'fast_lenght': 12, 'slow_lenght': 26, 'signal_lenght': 9}],
                     BBParams = [{'lenght': 20}],
                     OBVParams = [{'lenght': None}],
                     deTrendParams = {'column': 'Close', 'window': deTrendParams['optimalTrendSamples'], 'model': 'additive', 'weightModel': 'window_acorr', 'weightModelWindow': deTrendParams['optimalAcorrSamples']},
                     colPrefix = None,
                     dropNan = False,
                     force = True)

    df.to_csv(path_or_buf = savePath + '/' + asset + '_prep.CSV')
    if bot:
        message  = '{} deTrend completed at {}'.format(asset, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        bot.sendMessage(message)
