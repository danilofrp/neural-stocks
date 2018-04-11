# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..') #src directory
#sys.path.append('./src') #dev src directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #disables tensorflow 'not compilet to instruction X' messages
import time
import click
import pandas as pd
import multiprocessing
from neuralstocks.dataacquisition import *
from neuralstocks.models import RegressionSAE
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
@click.option('--loss', default = 'mse', help = 'Loss function to use for training. Default MSE')
@click.option('--optimizer', default = 'sgd', help = 'Optimizer alorithm to use for training. Default SGD')
@click.option('--force', is_flag = True, help = 'Forces new trainings even if previously trained models exist.')
@click.option('--verbose', is_flag = True, help = 'Verbosity flag.')
@click.option('--msg/--no-msg', default = False, help = 'Enables/disables telegram messaging. Defalut False')
@click.option('--dev', is_flag = True, help = 'Development flag, limits the number of datapoints to 400 and sets nInits to 1.')
def main(asset, inits, norm, loss, optimizer, force, verbose, msg, dev):
    bot = Bot('neuralStocks')
    dataPath, savePath = setPaths(__file__)
    savePath = savePath + '/diario/' + asset
    # dataPath = '../data'
    # savePath = '../ns-results/src/SAE/' + asset
    pathIBOV = dataPath + '/indexes/IBOV/diario/IBOV.CSV'
    pathUSDBRL = dataPath + '/forex/USDBRL/diario/USDBRL.CSV'
    pathAsset = savePath.replace('src/SAE', 'data/preprocessed') + '.CSV'
    # pathAsset = savePath.replace('src/SAE', 'data/preprocessed/diario') + '.CSV'

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

    # Creation SAE model:
    model = RegressionSAE(asset, savePath, dev)

    # training parameters
    hiddenLayers = [30, 20, 10]

    # Start Parallel processing
    initTime = datetime.now()
    minRMSE = model.train(X = xTrainNorm, y = yTrainNorm, hiddenLayers = hiddenLayers, nInits = inits,
                          loss = loss, optimizerAlgorithm = optimizer, force = force, verbose = verbose, dev = dev)

    finalModel = load_model(model.getSaveString(savePath +'/Models') + '.h5')
    predicted = yScaler.inverse_transform(finalModel.predict(xScaler.transform(xTest))).reshape(-1)
    predictedResid = pd.Series(predicted, index = df['2017'].index, name = '{}_resid_predicted_SAE_{}'.format(asset, norm))
    predictedSeries = pd.Series(df['{}_Close_trend'.format(asset)] +  predictedResid, name = '{}_Close_predicted_SAE_{}'.format(asset, norm))

    path = '{}{}{}'.format(pathAsset.split('preprocessed')[0], 'predicted/SAE/diario/', asset)
    filePath = '{}/{}_predicted_SAE{}.CSV'.format(path, asset, '_dev' if dev else '')
    if not os.path.exists(path) or not os.path.exists(filePath):
        try:
            os.makedirs(path)
        except OSError, e:
            if e.errno != os.errno.EEXIST:
                raise
            pass
        df = pd.concat([df[['{}_Close'.format(asset), '{}_Open'.format(asset), '{}_High'.format(asset), '{}_Low'.format(asset), '{}_Volume'.format(asset),
                            '{}_Close_trend'.format(asset), '{}_Close_resid'.format(asset)]], predictedResid, predictedSeries], axis = 1)
        df.to_csv(filePath)
    else:
        df2 = pd.read_csv(filePath, parse_dates=['Date'], index_col='Date').sort_index()
        df2.loc[:, predictedResid.name] = predictedResid
        df2.loc[:, predictedSeries.name] = predictedSeries
        df2.to_csv(filePath)

    if (msg):
        t = datetime.now() - initTime
        tStr = '{:02d}:{:02d}:{:02d}'.format(t.seconds//3600,(t.seconds//60)%60, t.seconds%60)
        message1 = '{} regression SAE ({}) training completed. Time elapsed: {}'.format(asset, norm, tStr)
        imgName = model.getSaveString(savePath +'/Figures', extra = 'fitHistory')
        try:
            bot.sendMessage([message1], imgPath = imgName + '.png', filePath = imgName + '.pdf')
        except Exception:
            pass


if __name__ == "__main__":
    main()
