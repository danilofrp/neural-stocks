# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..') #src directory
#sys.path.append('./src') #dev src directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disables tensorflow 'not compiled to instruction X' messages
import time
import click
import pandas as pd
import multiprocessing
from neuralstocks.dataacquisition import *
from neuralstocks.models import RegressionMLP
from neuralstocks import utils
from functools import partial
from sklearn.externals import joblib
from sklearn.model_selection import TimeSeriesSplit
from keras.models import load_model
from messaging.telegrambot import Bot
# </editor-fold>

asset = 'PETR4'
inits = 1
norm = 'mapminmax'

dataPath = '../data'
pathIBOV = dataPath + '/indexes/IBOV/diario/IBOV.CSV'
pathUSDBRL = dataPath + '/forex/USDBRL/diario/USDBRL.CSV'
pathAsset = savePath = '../ns-results/data/preprocessed/diario/' + asset + '.CSV'

# loading the (already preprocessed) data
ASSET = acquireData(filePath = pathAsset, dropNan = True)
IBOV = acquireData(filePath = pathIBOV, dropNan = True)
USDBRL= acquireData(filePath = pathUSDBRL, dropNan = True)
df = pd.concat([ASSET, IBOV, USDBRL], axis = 1).dropna()

# specifying which columns to use
columnsToUse = ['{}_Close_resid',
                '{}_Close_rollStd20',
                '{}_Close_returns', '{}_Close/Open_returns', '{}_High/Close_returns', '{}_Low/Close_returns',
                '{}_Close_EMA17_logdiff', '{}_Close_EMA72_logdiff', '{}_Close_EMA200_logdiff', '{}_Volume_EMA21_logdiff',
                '{}_MACD_12_26_9', '{}_MACDsignal_12_26_9', '{}_Bollinger%b_20', '{}_OBV',
                '{}_Holiday',
                'IBOV_Close_rollStd20',
                'IBOV_Close_returns', 'IBOV_Close/Open_returns', 'IBOV_High/Close_returns', 'IBOV_Low/Close_returns',
                'IBOV_Close_EMA17_logdiff', 'IBOV_Close_EMA72_logdiff', 'IBOV_Close_EMA200_logdiff',
                'USDBRL_Close_rollStd20',
                'USDBRL_Close_returns', 'USDBRL_Close/Open_returns', 'USDBRL_High/Close_returns', 'USDBRL_Low/Close_returns',
                'USDBRL_Close_EMA17_logdiff', 'USDBRL_Close_EMA72_logdiff', 'USDBRL_Close_EMA200_logdiff',
               ]
for i in range(len(columnsToUse)):
    columnsToUse[i] = columnsToUse[i].format(asset)

# creating the train and test sets
xTrain, yTrain, xTest, yTest = prepData(df = df, columnsToUse = columnsToUse,
                                        columnToPredict = '{}_Close_resid'.format(asset), nDelays = 10,
                                        testSetSize = len(df['2017'])#, validationSplitSize = 0.15
                                       )

xTrainNorm, xScaler = utils.normalizeData(xTrain, norm)
yTrainNorm, yScaler = utils.normalizeData(yTrain, norm)

%matplotlib inline
fig, ax = plotSeries(df['PETR4_Close'])


CVA = customTimeSeriesSplit(df, 6)

CVA[-1]['validation'].index[0]
CVA[-1]['validation'].index[-1]

saveDir = '/home/danilofrp/projeto_final/misc/imagens_junior'
plotTimeSeriesCVA(CVA, 'PETR4_Close', title = 'Time Series Splits for Cross Validation', saveImg = False, saveDir = saveDir, saveName = 'TS_split', saveFormat = 'png')

CVA2 = prepDataWithCrossValidation(df, ['PETR4_Close'], 'PETR4_Close', 10, 6)

































#.
