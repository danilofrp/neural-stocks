# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..') #src directory
sys.path.append('./src') #dev src directory
import multiprocessing
from functools import partial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralstocks.dataacquisition import *
from neuralstocks import MLP
from neuralstocks import utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from messaging.telegrambot import Bot
# </editor-fold>

stock = sys.argv[1]
norm = sys.argv[2]
# stock = 'PETR4'
if len(sys.argv) > 3 and sys.argv[3] == '--dev':
    dev = True
bot = Bot('neuralStocks')
dataPath, savePath = setPaths(__file__)
# dataPath = '../data'
# savePath = '../ns-results/src/MLP/' + stock
savePath = savePath + '/' + stock
pathIBOV = dataPath + '/indexes/IBOV/diario/IBOV.CSV'
pathUSDBRL = dataPath + '/forex/USDBRL/diario/USDBRL.CSV'
pathAsset = savePath.replace('src/MLP', 'data/preprocessed/diario') + '.CSV'

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
    columnsToUse[i] = columnsToUse[i].format(stock)

# creating the train and test sets
xTrain, yTrain, xTest, yTest = prepData(df = df, columnsToUse = columnsToUse,
                                        columnToPredict = '{}_Close_resid'.format(stock), nDelays = 10,
                                        testSetSize = len(df['2017'])#, validationSplitSize = 0.15
                                       )

xTrainNorm, xScaler = utils.normalizeData(xTrain, norm)
yTrainNorm, yScaler = utils.normalizeData(yTrain, norm)

# training parameters
nNeurons = range(1, xTrain.shape[1] + 1)

initTime = datetime.now()
# Start Parallel processing
func = partial(MLP.trainRegressionMLP, asset = stock, savePath = savePath, X = xTrainNorm, y = yTrainNorm, optimizer = 'Adam', dev = dev)
num_processes = multiprocessing.cpu_count()
p = multiprocessing.Pool(processes=num_processes)
results = p.map(func, nNeurons)
p.close()
p.join()

joblib.dump(results, '{}/{}/{}_modelsMSE{}.pkl'.format(savePath, 'Variables', stock, '_dev' if dev else ''))
bestModel = results.index(min(results)) + 1
joblib.dump(bestModel, '{}/{}/{}_bestModelNumberOfNeurons{}.pkl'.format(savePath, 'Variables', stock, '_dev' if dev else ''))

t = datetime.now() - initTime
tStr = '{:02d}:{:02d}:{:02d}'.format(t.seconds//3600,(t.seconds//60)%60, t.seconds%60)
message1 = '{} regression MLP training completed. Time elapsed: {}'.format(stock, tStr)
message2 = 'The best model had {} neurons in the hidden layer.'.format(bestModel)
imgName = utils.getSaveString(savePath +'/Figures', stock, 'MLP', xTrain.shape[1], bestModel, norm, extra = 'fitHistory', dev = dev)
bot.sendMessage([message1, message2], imgPath = imgName + '.png', filePath = imgName + '.pdf')
