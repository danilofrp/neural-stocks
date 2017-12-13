# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyTaLib.indicators import *
from neuralstocks.dataacquisition import *
from neuralstocks.preprocessing import *
from neuralstocks.plots import *
from neuralstocks.utils import *
%matplotlib inline
# </editor-fold>

# <editor-fold> PARAMS
dataPath = '/home/danilofrp/projeto_final/data'
assetType = 'stocks'
asset = 'PETR4'
frequency = 'diario'

filePath = dataPath + '/' + assetType + '/' + asset + '/' + frequency + '/' + asset + '.CSV'

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

df = acquireData(filePath = filePath,
                 replicateForHolidays = True,
                 meanStdLen = 20,
                 returnCalcParams = [['Close'], ['Close', 'Open']],
                 EMAparams = [{'column': 'Close', 'lenght': 17}, {'column': 'Close', 'lenght': 72}, {'column': 'Volume', 'lenght': 21}],
                 MACDParams = [{'fast_lenght': 12, 'slow_lenght': 26, 'signal_lenght': 9}],
                 BBParams = [{'lenght': 20}],
                 OBVParams = [{'lenght': None}],
                 #deTrendParams = {'column': 'Close', 'window': 10, 'model': decomposeModel, 'weightModel': 'window_acorr', 'weightModelWindow': 25},
                 colPrefix = None,
                 dropNan = False)
df.tail(1)
df.columns.values

plotSeries([df['Close'], df['Close_trend']], title = None, initialPlotDate = '2017', finalPlotDate = '2017', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotReturnSeries(df, column = 'Close', asset = asset,  initialPlotDate = '', finalPlotDate = '', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = None, weightModelWindow = 25,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, detailed = True, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weightModel = 'full_pgram', weightModelWindow = 25,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, detailed = True, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrend(df, column = 'Close', window = 25, model = decomposeModel, fitOrder = 1, weightModel = 'window_pgram', weightModelWindow = 100,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, detailed = True, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrend(df, column = 'Close', window = 10, model = decomposeModel, fitOrder = 1, weightModel = 'window_acorr', weightModelWindow = 25,
            plot = True, initialPlotDate = '', finalPlotDate = '', overlap = True, detailed = True, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deSeason(df, column = 'Close', freq = 5, model = decomposeModel, plot = True, initialPlotDate = '2017', finalPlotDate = '2017', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 10, weightModel = None, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weightModel = 'full_pgram', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrendRMSE(df[:'2016'], column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 25, weightModel = 'window_pgram', weightModelWindow = 100, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deTrendRMSE(df, column = 'Close', model = decomposeModel, fitOrder = 1, windowMaxSize = 10, weightModel = 'window_acorr', weightModelWindow = 10, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

deSeasonRMSE(df, column = 'Close', model = decomposeModel, maxFreq = 75, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

decompose(df, column = 'Close', model = decomposeModel, window = 3, freq = 5, weightModel = 'window_acorr', weightModelWindow = None,
          plot = False, initialPlotDate = '2008', finalPlotDate = '2008', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotPeriodogramStats(df['Close_EMA72_logdiff'], plotInit = 2, plotEnd = 100, yLog = False, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotPeriodogramSciPy(df['Close_EMA72_logdiff'], plotInit = 2, plotEnd = 100, yLog = False, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotFFT(df['Close_EMA72_logdiff'], saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotSeasonalDecompose(df['Close'],  asset = asset, frequency=5, initialPlotDate='2016', finalPlotDate='2017', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

testStationarity(df['Close_resid'], window=25, initialPlotDate='', finalPlotDate='', saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotAcf(df['Close'][:4000][-75:], lags = 25, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

plotCrosscorrelation(df['Close_returns'], df['Close_EMA72_logdiff'], 50, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

histogram([df['Close'], df['Close_trend']], colors = ['b', 'r'], nBins=100, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
deTrend(df, column = 'Close', window = 3, model = decomposeModel, fitOrder = 1, weightModel = 'full_acorr', weightModelWindow = None)
ax.plot(df['Close_trend'], df['Close'], 'bo')
deTrend(df, column = 'Close', window = 10, model = decomposeModel, fitOrder = 1, weightModel = 'window_acorr', weightModelWindow = 25)
ax.plot(df['Close_trend'], df['Close'], 'ro', alpha=0.5)

scatterHist(df['Close'], df['Close_trend'], nBins = 100, saveImg = False, saveDir = saveDir, saveName = '', saveFormat = saveFormat)

# </editor-fold>

# <editor-fold> MISC
df2 = df.copy()
windowMaxSize = 10
weightModelWindowMaxSize = 50
model = 'additive'
column = 'Close'
minimal = None
RMSE = np.empty((windowMaxSize + 1, weightModelWindowMaxSize + 1), dtype=float)*np.nan
for i in range(2, windowMaxSize + 1):
    for j in range(2 * i, weightModelWindowMaxSize + 1):
        print('Running deTrend ({}, {})'.format(i, j), end='\r')
        deTrend(df2, column = column, window = i, model = model, fitOrder = 1, weightModel = 'window_acorr', weightModelWindow = j)
        if model.startswith('m'):
            RMSE[i, j] = np.square((df2['{}_resid'.format(column)] - 1)).sum()/len(df2['{}_resid'.format(column)].dropna())
        else:
            RMSE[i, j] = np.square(df2['{}_resid'.format(column)]).sum()/len(df2['{}_resid'.format(column)].dropna())
        if not minimal:
            minimal = RMSE[i][j]
            iMin = i
            jMin = j
        if RMSE[i, j] < minimal:
            minimal = RMSE[i][j]
            iMin = i
            jMin = j

print('Minimal RSME = {}, at window = {} and weightModelWindow = {} * window'.format(minimal, iMin, jMin))

#RMSE_backup = RMSE
fig, ax = plt.subplots(figsize=(10,10))
cax = ax.imshow(RMSE[:,:], cmap="jet", aspect="auto")
plt.gca().invert_yaxis()
ax.set_title('RMSE analysis: ACF sweep x Trend sweep')
ax.set_ylim([3.5,10.5])
ax.set_xlim([5.5,50.5])
ax.set_xlabel('Delayed samples used for ACF estimation')
ax.set_ylabel('Delayed samples used for trend estimation')
cbar = fig.colorbar(cax)
plt.figtext(0.5,  0.010, 'Minimal RSME: {:.3f}, using Trend window with {} samples \nand ACF window with {} samples'.format(minimal, iMin, jMin), size = 14, horizontalalignment = 'center')
fig.savefig('{}/{}.{}'.format(saveDir, 'PETR4_RMSEanalysis_acf', 'pdf'), bbox_inches='tight')



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



df4 = df.copy()
windowMaxSize = 20
maxFreq = 30
model = 'additive'
column = 'Close'
RSS = np.empty((windowMaxSize + 1, maxFreq + 1), dtype=float)*0
for i in range(2, windowMaxSize + 1):
    deTrend(df4, column = column, window = i, model = model, fitOrder = 1)
    for j in range(maxFreq + 1):
        deSeason(df4, column = column, freq = j, model = model)
        if model == 'multiplicative':
            RSS[i, j] = np.square((df4['{}_resid'.format(column)] - 1)).sum()
        else:
            RSS[i, j] = np.square(df4['{}_resid'.format(column)]).sum()

fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(RSS[2:,:], cmap="jet", extent=[2, windowMaxSize, 0, maxFreq], aspect="auto")
cbar = plt.colorbar()
# </editor-fold>
