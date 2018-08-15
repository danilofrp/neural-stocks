# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyTaLib.indicators import *
from neuralstocks.dataacquisition import *
from neuralstocks.preprocessing import *
from neuralstocks.plots import *
from neuralstocks.utils import *
from neuralstocks.backtesting import Backtest
%matplotlib inline
# </editor-fold>

# <editor-fold> Functions
def plotDecompose(df, column, figsize = (8,10), saveImg = False, saveDir = '', saveName = ''):
    df = df.dropna()
    trendName = column + '_trend'
    residName = column + '_resid'
    plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(top=1, bottom=0.0)
    ax1 = plt.subplot(gs1[0, :])
    ax1.set_title('Observado')
    ax1.set_ylabel('BRL')
    ax1.plot(df[column])
    ax1.set_xticklabels([])
    ax1.grid()

    ax2 = plt.subplot(gs1[1, :])
    ax2.set_title(u'Tendência Estimada')
    ax2.set_ylabel('BRL')
    ax2.plot(df[trendName])
    ax2.set_xticklabels([])
    ax2.grid()

    ax3 = plt.subplot(gs1[2, :])
    ax3.set_title(u'Série Residual')
    ax3.set_ylabel('Erro (BRL)')
    ax3.plot(df[residName])
    ax3.set_xlabel('Date')
    ax3.grid()

    if saveImg:
        saveName = saveName if saveName else 'decmpose'
        fig.savefig('{}/{}.png'.format(saveDir, 'decompose'), bbox_inches='tight')
        fig.savefig('{}/{}.pdf'.format(saveDir, 'decompose'), bbox_inches='tight')

def plotPredictionsHistogram(asset, saveImg = False, saveDir = ''):
    path_MLP_class = '/home/danilofrp/projeto_final/ns-results/data/predicted/MLP_class/diario/{}/{}_bin_predicted_MLP.CSV'.format(asset, asset)
    path_SAE_class = '/home/danilofrp/projeto_final/ns-results/data/predicted/SAE_class/diario/{}/{}_bin_predicted_SAE.CSV'.format(asset, asset)
    MLP_class = pd.read_csv(path_MLP_class, parse_dates=['Date'], index_col='Date')[['{}_Close/Open_returns'.format(asset), '{}_bin_predicted_MLP_mapminmax'.format(asset)]].sort_index()
    SAE_class = pd.read_csv(path_SAE_class, parse_dates=['Date'], index_col='Date')['{}_bin_predicted_SAE_mapminmax'.format(asset)].sort_index()
    df = pd.concat([MLP_class, SAE_class], axis = 1)

    mlp_positive_returns = df[df['{}_Close/Open_returns'.format(asset)] > 0]['2017']['{}_bin_predicted_MLP_mapminmax'.format(asset)].values
    mlp_negative_returns = df[df['{}_Close/Open_returns'.format(asset)] < 0]['2017']['{}_bin_predicted_MLP_mapminmax'.format(asset)].values
    sae_positive_returns = df[df['{}_Close/Open_returns'.format(asset)] > 0]['2017']['{}_bin_predicted_SAE_mapminmax'.format(asset)].values
    sae_negative_returns = df[df['{}_Close/Open_returns'.format(asset)] < 0]['2017']['{}_bin_predicted_SAE_mapminmax'.format(asset)].values
    bins = np.linspace(-0.5, 0.5, 30)

    fig, ax = plt.subplots(figsize=(16,8), nrows = 1, ncols = 2)

    ax[0].hist(mlp_positive_returns, bins, alpha=0.5, label='Retornos Positivos')
    ax[0].hist(mlp_negative_returns, bins, alpha=0.5, label='Retornos Negativos')
    ax[0].set_title('MLP: Histograma de classes previstas')
    ax[0].set_xlabel(u'Saída da Rede')
    ax[0].set_ylabel(u'# de Ocorrências')
    ax[0].legend(prop={'size': 15})

    ax[1].hist(sae_positive_returns, bins, alpha=0.5, label='Retornos Positivos')
    ax[1].hist(sae_negative_returns, bins, alpha=0.5, label='Retornos Negativos')
    ax[1].set_title('SAE: Histograma de classes previstas')
    ax[1].set_xlabel(u'Saída da Rede')
    ax[1].set_ylabel(u'# de Ocorrências')
    ax[1].legend(prop={'size': 15})

    if saveImg:
        fig.savefig('{}/{}.png'.format(saveDir, 'bin_predicted_hist'), bbox_inches='tight')
        fig.savefig('{}/{}.pdf'.format(saveDir, 'bin_predicted_hist'), bbox_inches='tight')

def plotBacktest(asset, saveImg, saveDir):
    path_MLP = '/home/danilofrp/projeto_final/ns-results/data/predicted/MLP/diario/{}/{}_predicted_MLP.CSV'.format(asset, asset)
    path_SAE = '/home/danilofrp/projeto_final/ns-results/data/predicted/SAE/diario/{}/{}_predicted_SAE.CSV'.format(asset, asset)
    path_MLP_class = '/home/danilofrp/projeto_final/ns-results/data/predicted/MLP_class/diario/{}/{}_bin_predicted_MLP.CSV'.format(asset, asset)
    path_SAE_class = '/home/danilofrp/projeto_final/ns-results/data/predicted/SAE_class/diario/{}/{}_bin_predicted_SAE.CSV'.format(asset, asset)

    MLP = pd.read_csv(path_MLP, parse_dates=['Date'], index_col='Date')[['{}_Close_trend'.format(asset),'{}_Close_predicted_MLP_mapminmax'.format(asset)]].sort_index()
    SAE = pd.read_csv(path_SAE, parse_dates=['Date'], index_col='Date')['{}_Close_predicted_SAE_mapminmax'.format(asset)].sort_index()
    MLP_class = pd.read_csv(path_MLP_class, parse_dates=['Date'], index_col='Date')['{}_bin_predicted_MLP_mapminmax'.format(asset)].sort_index()
    SAE_class = pd.read_csv(path_SAE_class, parse_dates=['Date'], index_col='Date')['{}_bin_predicted_SAE_mapminmax'.format(asset)].sort_index()
    df = pd.concat([MLP, SAE, MLP_class, SAE_class], axis = 1)

    assetsToUse = [asset]
    bt = Backtest(assets = assetsToUse, dataPath = '/home/danilofrp/projeto_final/data/stocks/[asset]/diario/[asset].CSV')
    initialFunds = 100000
    startDate = '2017'
    endDate = '2017'

    bt.simulate(strategy = 'buy-n-hold', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse, verbose = 0)

    bt.simulate(strategy = 'predicted', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                predicted = {asset: df['{}_Close_trend'.format(asset)]}, simulationName = 'predicted_reg_trend', verbose = 0)

    bt.simulate(strategy = 'predicted', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                predicted = {asset: df['{}_Close_predicted_MLP_mapminmax'.format(asset)]}, simulationName = 'predicted_reg_mlp', verbose = 0)

    bt.simulate(strategy = 'predicted', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                predicted = {asset: df['{}_Close_predicted_SAE_mapminmax'.format(asset)]}, simulationName = 'predicted_reg_sae', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_MLP_mapminmax'.format(asset)]}, simulationName = 'predicted_bin_mlp', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_SAE_mapminmax'.format(asset)]}, simulationName = 'predicted_bin_sae', verbose = 0)

    bt.plotSimulations(simulations = ['buy-n-hold', 'predicted_reg_trend', 'predicted_reg_mlp', 'predicted_reg_sae'],
                       names = ['Buy\'n\'Hold', u'Tendência', u'Tendência + MLP', u'Tendência + SAE'], locale = 'pt',
                       legendsize = 16, linewidth = 4.0, legendPos = (-0.9, -0.4), legendncol = 4,
                       saveImg = saveImg, saveDir = saveDir, saveName = 'backtest_reg_pt')

    bt.plotSimulations(simulations = ['buy-n-hold', 'predicted_bin_mlp', 'predicted_bin_sae'],
                       names = ['Buy\'n\'Hold', 'MLP (Class.)', 'SAE (Class.)'], locale = 'pt',
                       legendsize = 16, linewidth = 4.0, legendPos = (-0.62, -0.4), legendncol = 3,
                       saveImg = saveImg, saveDir = saveDir, saveName = 'backtest_bin_pt')

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_MLP_mapminmax'.format(asset)]}, confidenceLimit = 0.00, simulationName = 'predicted_bin_mlp_0.00', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_MLP_mapminmax'.format(asset)]}, confidenceLimit = 0.05, simulationName = 'predicted_bin_mlp_0.05', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_MLP_mapminmax'.format(asset)]}, confidenceLimit = 0.10, simulationName = 'predicted_bin_mlp_0.10', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_MLP_mapminmax'.format(asset)]}, confidenceLimit = 0.15, simulationName = 'predicted_bin_mlp_0.15', verbose = 0)

    bt.plotSimulations(simulations = ['buy-n-hold', 'predicted_bin_mlp_0.00', 'predicted_bin_mlp_0.05', 'predicted_bin_mlp_0.10', 'predicted_bin_mlp_0.15'],
                       names = ['Buy\'n\'Hold', r'$\lambda$ = 0.00', r'$\lambda$ = 0.05', r'$\lambda$ = 0.10', r'$\lambda$ = 0.15'], locale = 'pt',
                       legendsize = 16, linewidth = 4.0, legendPos = (-0.85 , -0.4), legendncol = 5,
                       saveImg = saveImg, saveDir = saveDir, saveName = 'backtest_bin_mlpLim_pt')

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_SAE_mapminmax'.format(asset)]}, confidenceLimit = 0.00, simulationName = 'predicted_bin_sae_0.00', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_SAE_mapminmax'.format(asset)]}, confidenceLimit = 0.05, simulationName = 'predicted_bin_sae_0.05', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_SAE_mapminmax'.format(asset)]}, confidenceLimit = 0.10, simulationName = 'predicted_bin_sae_0.10', verbose = 0)

    bt.simulate(strategy = 'orders', start = startDate, end = endDate, initialFunds = initialFunds, assetsToUse = assetsToUse,
                orders = {asset: df['{}_bin_predicted_SAE_mapminmax'.format(asset)]}, confidenceLimit = 0.15, simulationName = 'predicted_bin_sae_0.15', verbose = 0)

    bt.plotSimulations(simulations = ['buy-n-hold', 'predicted_bin_sae_0.00', 'predicted_bin_sae_0.05', 'predicted_bin_sae_0.10', 'predicted_bin_sae_0.15'],
                       names = ['Buy\'n\'Hold', r'$\lambda$ = 0.00', r'$\lambda$ = 0.05', r'$\lambda$ = 0.10', r'$\lambda$ = 0.15'], locale = 'pt',
                       legendsize = 16, linewidth = 4.0, legendPos = (-0.85 , -0.4), legendncol = 5,
                       saveImg = saveImg, saveDir = saveDir, saveName = 'backtest_bin_saeLim_pt')

# </editor-fold>

# <editor-fold> RCPARAMS
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# </editor-fold>

assets = ['ABEV3', 'BRFS3', 'BVMF3', 'CCRO3', 'ELET3', 'ITUB4', 'KROT3', 'LAME4', 'PETR4', 'SUZB5', 'USIM5', 'VALE5', 'VIVT4']
#asset = 'PETR4'
for asset in assets:
    filePath = '/home/danilofrp/projeto_final/data/stocks/' + asset + '/diario/' + asset + '.CSV'
    saveDir = '/home/danilofrp/projeto_final/docs/projeto_final/figuras/ApendiceA/' + asset + '/'
    if not os.path.exists(saveDir): os.makedirs(saveDir)

    # <editor-fold> Data Aquisition
    df = acquireData(filePath = filePath,  dropNan = False)
    # </editor-fold>

    plotSeries([df['{}_Close'.format(asset)]], title = u'Preço de Fechamento', ylabel = 'BRL', initialPlotDate = '', finalPlotDate = '', legend = False, saveImg = True, saveDir = saveDir, saveName = 'historic_close', saveFormat = 'pdf')

    plotSeries([df['{}_Close_returns'.format(asset)]], title = u'Log-Retornos do Preço de Fechamento', ylabel = 'BRL', initialPlotDate = '', finalPlotDate = '', legend = True, saveImg = True, saveDir = saveDir, saveName = 'close_log_returns', saveFormat = 'pdf')

    plotDecompose(df, '{}_Close'.format(asset), figsize = (12,8), saveImg = True, saveDir = saveDir)

    plotPredictionsHistogram(asset, saveImg = True, saveDir = saveDir)

    plotBacktest(asset, saveImg = True, saveDir = saveDir)




















##### CAP5 ######

# <editor-fold> PARAMS
baseDir = '/home/danilofrp/projeto_final'
dataPath = baseDir + '/data'
asset = 'PETR4'

filePath = dataPath + '/stocks/' + asset + '/diario/' + asset + '.CSV'
ibovPath = dataPath + '/indexes/IBOV/diario/IBOV.CSV'
usdbrlPath = dataPath + '/forex/USDBRL/diario/USDBRL.CSV'

saveDir = '/home/danilofrp/projeto_final/docs/projeto_final/figuras/cap5/'
if not os.path.exists(saveDir): os.makedirs(saveDir)

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# </editor-fold>

# <editor-fold> acquire data
df = acquireData(filePath = filePath,
                 replicateForHolidays = True,
                 colPrefix = None,
                 dropNan = False)
df_ibov = acquireData(filePath = ibovPath,
                 replicateForHolidays = True,
                 colPrefix = None,
                 dropNan = False)
df_usdbrl = acquireData(filePath = usdbrlPath,
                 replicateForHolidays = True,
                 colPrefix = None,
                 dropNan = False)
df = pd.concat([df, df_ibov, df_usdbrl], axis = 1).dropna()
df.tail(1)
df.columns
# </editor-fold>

# <editor-fold> close series
plotAsset = 'IBOV'
fig, ax = plt.subplots(figsize=(10,5), nrows = 1, ncols = 1)
fig.suptitle(u'Cotação de Fechamento')
ax.set_xlabel('Data')
ax.set_ylabel('Pontos')
ax.plot(df['{}_Close'.format(plotAsset)])
ax.grid()
fig.autofmt_xdate()

fig.savefig('{}/{}.pdf'.format(saveDir, '{}_Close'.format(plotAsset)), bbox_inches='tight')
fig.savefig('{}/{}.png'.format(saveDir, '{}_Close'.format(plotAsset)), bbox_inches='tight')

# </editor-fold>

# <editor-fold> linear fit
window = 6
acorr_window = 17
offset = 4100

x = range(1, window + 1)
y = df['PETR4_Close'].values[offset : offset + window]

a_no_weights = np.polyfit(x, y, deg = 1)
y_pred_no_weights = [(a_no_weights[0]*i + a_no_weights[1]) for i in range(1, window + 2)]

weights = list(reversed(autocorrelation(df['PETR4_Close'][offset - acorr_window + window : offset + window], nlags = acorr_window)[1:window + 1]))
weights = [abs(w) for w in weights]
a_weights = np.polyfit(x, y, deg = 1, w = weights)
y_pred_weights = [(a_weights[0]*i + a_weights[1]) for i in range(1, window + 2)]

fig, ax = plt.subplots(figsize=(8,8), nrows = 1, ncols = 1)
ax.grid()
# fig.suptitle('Trend Estimations')
ax.set_ylim((12, 13.2))
ax.set_title(u'Estimação da Tendência', fontsize = 20)
ax.set_xlabel('Amostra')
ax.set_ylabel('BRL')
h1 = ax.scatter(x, y, color = 'indigo', label = 'Dados Passados', s = 75)
h2 = ax.scatter(window + 1, df['PETR4_Close'][offset + window], color = 'orangered', label = 'Valor Futuro', s = 100, marker = '^')

h3, = ax.plot(range(1, window + 1), y_pred_no_weights[:-1], color = 'crimson', label = 'Fit Linear', linewidth = 4)
ax.plot(range(1, window + 2), y_pred_no_weights, color = 'crimson', linestyle = '--', linewidth = 4)

h4, = ax.plot(range(1, window + 1), y_pred_weights[:-1], color = 'green', label = 'Fit Linear Pesado', linewidth = 4)
ax.plot(range(1, window + 2), y_pred_weights, color = 'green', linestyle = '--', linewidth = 4)

handles = [h1, h2, h3, h4]
labels = ['Dados Passados', 'Valor Futuro', 'Fit Linear', 'Fit Linear Pesado']
plt.legend(handles, labels, prop={'size': 14})
# fig.autofmt_xdate()

fig.savefig('{}/{}.{}'.format(saveDir, 'deTrend_sample_prediction_2', 'pdf'), bbox_inches='tight')
fig.savefig('{}/{}.{}'.format(saveDir, 'deTrend_sample_prediction_2', 'png'), bbox_inches='tight')

# </editor-fold>

# <editor-fold> autocorrelation
acorr = autocorrelation(df['PETR4_Close'][offset - acorr_window + window : offset + window], nlags = acorr_window)[1:window + 1]

fig, ax = plt.subplots(figsize=(6.5,4), nrows = 1, ncols = 1)
ax.grid()
ax.set_title(u'Autocorrelação estimada', fontsize = 20)
ax.set_xlabel('# de atrasos')
markerline, stemlines, baseline = ax.stem(range(1, window + 1), acorr)
plt.setp(baseline, color='k')
plt.setp(stemlines, linewidth = 3)
plt.setp(markerline, markersize = 10)

fig.savefig('{}/{}.{}'.format(saveDir, 'estimated_acorr', 'pdf'), bbox_inches='tight')
fig.savefig('{}/{}.{}'.format(saveDir, 'estimated_acorr', 'png'), bbox_inches='tight')
# </editor-fold>

# <editor-fold> detrend RMSE
from neuralstocks import preprocessing
asset = 'VIVT4'
filePath = '/home/danilofrp/projeto_final/data/stocks/' + asset + '/diario/' + asset + '.CSV'
saveDir = '/home/danilofrp/projeto_final/docs/projeto_final/figuras/ApendiceA/' + asset
if not os.path.exists(saveDir): os.makedirs(saveDir)
df = acquireData(filePath = filePath,  dropNan = False)
preprocessing.deTrendRMSE(df[:'2016'], column = '{}_Close'.format(asset), model = 'additive', fitOrder = 1, windowMaxSize = 15, figsize = (8,8), weightModel = None, saveImg = True, saveDir = saveDir, saveName = 'deTrend_RMSE', locale = 'pt')

from sklearn.externals import joblib
windowMinSize = 3
weightModelWindowMinSize = 6
windowMaxSize = 25
weightModelWindowMaxSize = 150
RMSE = joblib.load('/home/danilofrp/projeto_final/ns-results/src/deTrend/Variables/{}_RMSEanalysis_acf_w25m150.pkl'.format(asset))
minimal = np.Inf
for i in range(windowMinSize, windowMaxSize + 1):
    for j in range(2 * i, weightModelWindowMaxSize + 1):
        if RMSE[i, j] < minimal:
            minimal = RMSE[i][j]
            iMin = i
            jMin = j

fig, ax = plt.subplots(figsize=(8,8))
cax = ax.imshow(RMSE[:,:], cmap='jet', aspect='auto')
plt.gca().invert_yaxis()
ax.set_title(u'Análise de RMSE: Busca paramétrica')
ax.set_ylim([windowMinSize + 0.5, windowMaxSize + 0.5])
ax.set_xlim([weightModelWindowMinSize + 0.5, weightModelWindowMaxSize + 0.5])
ax.set_xlabel(u'Amostras passadas usadas para\n a estimação da autocorrelação')
ax.set_ylabel(u'Amostras passadas usadas para\n a estimação da tendência')
cbar = fig.colorbar(cax)
plt.figtext(0.45,  -0.1, u'RMSE Mínimo: {:.5f}, com {} amostras para a tendência \ne {} amostras para a autocorrelação'.format(minimal, iMin, jMin), size = 16, horizontalalignment = 'center')

fig.savefig('{}/{}.{}'.format(saveDir, 'deTrend_acorr_RMSE', 'pdf'), bbox_inches='tight')
fig.savefig('{}/{}.{}'.format(saveDir, 'deTrend_acorr_RMSE', 'png'), bbox_inches='tight')
# </editor-fold>

# <editor-fold> FFT
assets = ['ABEV3', 'BRFS3', 'BVMF3', 'CCRO3', 'ELET3', 'ITUB4', 'KROT3', 'LAME4', 'PETR4', 'SUZB5', 'USIM5', 'VALE5', 'VIVT4']
for asset in assets:
    filePath = '/home/danilofrp/projeto_final/data/stocks/' + asset + '/diario/' + asset + '.CSV'
    saveDir = '/home/danilofrp/projeto_final/docs/projeto_final/figuras/ApendiceA/' + asset
    df = acquireData(filePath = filePath,  dropNan = False)
    y = df['{}_Close_resid'.format(asset)]
    Fs = 1.0;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    n = len(y) # lenght of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]

    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.set_yscale('log')
    ax.plot(frq, abs(Y)) # plotting the spectrum
    ax.set_xlabel('Freq')
    ax.set_ylabel('|X(freq)|')
    ax.set_title('FFT')

    fig.savefig('{}/{}.{}'.format(saveDir, 'resid_FFT', 'pdf'), bbox_inches='tight')
    fig.savefig('{}/{}.{}'.format(saveDir, 'resid_FFT', 'png'), bbox_inches='tight')
# </editor-fold>
























##
