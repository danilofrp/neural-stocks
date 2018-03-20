# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..') #src directory
#sys.path.append('./src') #dev src directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disables tensorflow 'not compilet to instruction X' messages
import time
import click
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
from neuralstocks.dataacquisition import *
from neuralstocks import MLP
from neuralstocks import utils
from functools import partial
from sklearn.externals import joblib
from keras.models import load_model
from messaging.telegrambot import Bot
# </editor-fold>

@click.command()
@click.option('--asset', help = 'Asset to run the analysis on.')
@click.option('--inits', default = 1, help = 'Number of initializations for a single neural network topology. Default 1')
@click.option('--norm', default = 'mapminmax', help = 'Normalization technique to use. Default mapminmax')
@click.option('--optimizer', default = 'sgd', help = 'Optimizer alorithm to use for training. Default SGD')
@click.option('--verbose', is_flag = True, help = 'Verbosity flag.')
@click.option('--msg/--no-msg', default = False, help = 'Disable telegram messaging.')
@click.option('--dev', is_flag = True, help = 'Development flag, limits the number of datapoints to 400 and sets nInits to 1.')
def main(asset, inits, norm, optimizer, verbose, msg, dev):
    bot = Bot('neuralStocks')
    dataPath, savePath = setPaths(__file__)
    # dataPath = '../data'
    # savePath = '../ns-results/src/MLP/' + asset
    savePath = savePath + '/diario/' + asset
    pathIBOV = dataPath + '/indexes/IBOV/diario/IBOV.CSV'
    pathUSDBRL = dataPath + '/forex/USDBRL/diario/USDBRL.CSV'
    pathAsset = savePath.replace('src/MLP', 'data/preprocessed') + '.CSV'

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

    # training parameters
    nNeurons = range(1, xTrain.shape[1] + 1)

    initTime = datetime.now()
    # Start Parallel processing
    func = partial(MLP.trainRegressionMLP, asset = asset, savePath = savePath, X = xTrainNorm, y = yTrainNorm,
                   norm = norm, nInits = inits, optimizerAlgorithm = optimizer, verbose = verbose, dev = dev)
    num_processes = multiprocessing.cpu_count()
    p = multiprocessing.Pool(processes=num_processes)
    results = p.map(func, nNeurons)
    p.close()
    p.join()

    joblib.dump(results, '{}/{}/{}_modelsMSE{}.pkl'.format(savePath, 'Variables', asset, '_dev' if dev else ''))
    bestModelNumberOfNeurons = results.index(min(results)) + 1
    joblib.dump(bestModelNumberOfNeurons, '{}/{}/{}_bestModelNumberOfNeurons{}.pkl'.format(savePath, 'Variables', asset, '_dev' if dev else ''))

    bestModel = load_model(utils.getSaveString(savePath +'/Models', asset, 'regression_MLP', xTrain.shape[1], bestModelNumberOfNeurons, optimizer, norm, dev = dev) + '.h5')
    predicted = yScaler.inverse_transform(bestModel.predict(xScaler.transform(xTest))).reshape(-1)
    predictedResid = pd.Series(predicted, index = df['2017'].index, name = '{}_resid_predicted_MLP_{}'.format(asset, norm))
    predictedSeries = pd.Series(df['{}_Close_trend'.format(asset)] +  predictedResid, name = '{}_Close_predicted_MLP_{}'.format(asset, norm))

    plotSeries([df['{}_Close'.format(asset)], df['{}_Close_trend'.format(asset)], predictedSeries],
               initialPlotDate = '2017', finalPlotDate = '2017',
               title = 'Original Data vs Predicted', ylabel = 'Price',
               saveImg = True, saveFormat = 'pdf',
               savePath = '{}/{}_{}_{}_{}{}'.format(savePath + '/Figures', asset, 'regression_MLP', norm, 'predicted',  '_dev' if dev else ''))

    df = pd.concat([df[['{}_Close'.format(asset), '{}_Open'.format(asset), '{}_High'.format(asset), '{}_Low'.format(asset), '{}_Volume'.format(asset),
                        '{}_Close_trend'.format(asset), '{}_Close_resid'.format(asset)]], predictedResid, predictedSeries], axis = 1)
    path = '{}{}{}'.format(pathAsset.split('preprocessed')[0], 'predicted/MLP/diario/', asset)
    if not os.path.exists(path): os.makedirs(path)
    df.to_csv('{}{}/{}/{}_predicted_MLP_{}{}.CSV'.format(pathAsset.split('preprocessed')[0], 'predicted/MLP/diario', asset, asset, norm, '_dev' if dev else ''))

    if (msg):
        t = datetime.now() - initTime
        tStr = '{:02d}:{:02d}:{:02d}'.format(t.seconds//3600,(t.seconds//60)%60, t.seconds%60)
        message1 = '{} regression MLP training completed. Time elapsed: {}'.format(asset, tStr)
        message2 = 'The best model had {} neurons in the hidden layer.'.format(bestModelNumberOfNeurons)
        imgName = utils.getSaveString(savePath +'/Figures', asset, 'regression_MLP', xTrain.shape[1], bestModelNumberOfNeurons, optimizer, norm, extra = 'fitHistory', dev = dev)
        try:
            bot.sendMessage([message1, message2], imgPath = imgName + '.png', filePath = imgName + '.pdf')
        except Exception:
            pass


if __name__ == "__main__":
    main()
