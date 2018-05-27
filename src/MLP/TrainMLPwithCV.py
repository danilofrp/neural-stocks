# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
from neuralstocks.dataacquisition import *
from neuralstocks.models import RegressionMLP
from neuralstocks import utils
from functools import partial
from sklearn.externals import joblib
from keras.models import load_model
from messaging.telegrambot import Bot
# </editor-fold>

def trainWrapper(neurons, model, CVA, nInits, epochs, validationSplit, loss, optimizerAlgorithm, verbose, dev):
    return model.trainWithCrossValidation(CVA, hiddenLayers = neurons, nInits = nInits, epochs = epochs,
                       validationSplit = validationSplit, loss = loss, optimizerAlgorithm = optimizerAlgorithm,
                       verbose = verbose, dev = dev)

@click.command()
@click.option('--asset', help = 'Asset to run the analysis on.')
@click.option('--inits', default = 1, help = 'Number of initializations for a single neural network topology. Default 1')
@click.option('--norm', default = 'mapminmax', help = 'Normalization technique to use. Default mapminmax')
@click.option('--loss', default = 'mse', help = 'Loss function to use for training. Default MSE')
@click.option('--optimizer', default = 'sgd', help = 'Optimizer alorithm to use for training. Default SGD')
@click.option('--verbose', is_flag = True, help = 'Verbosity flag.')
@click.option('--msg/--no-msg', default = False, help = 'Enables/disables telegram messaging. Defalut False')
@click.option('--dev', is_flag = True, help = 'Development flag, limits the number of datapoints to 400 and sets nInits to 1.')
def main(asset, inits, norm, loss, optimizer, verbose, msg, dev):
    bot = Bot('neuralStocks')
    dataPath, savePath = setPaths(__file__)
    savePath = savePath + '/diario/' + asset
    # dataPath = '../data'
    # savePath = '../ns-results/src/MLP/' + asset
    pathIBOV = dataPath + '/indexes/IBOV/diario/IBOV.CSV'
    pathUSDBRL = dataPath + '/forex/USDBRL/diario/USDBRL.CSV'
    pathAsset = savePath.replace('src/MLP', 'data/preprocessed') + '.CSV'
    # pathAsset = savePath.replace('src/MLP', 'data/preprocessed/diario') + '.CSV'

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
    CVA = prepDataWithCrossValidation(df = df, columnsToUse = columnsToUse,
                                            columnToPredict = '{}_Close_resid'.format(asset),
                                            nDelays = 10, nSplits = 6
                                           )

    _, xScaler = utils.normalizeData(CVA[-1]['x_train'], norm)
    _, yScaler = utils.normalizeData(CVA[-1]['y_train'], norm)
    CVAnorm = []
    for CVO in CVA:
        CVAnorm.append({
                        'x_train': utils.normalizeData(CVO['x_train'], norm, xScaler),
                        'y_train': utils.normalizeData(CVO['y_train'], norm, yScaler),
                        'x_validation': utils.normalizeData(CVO['x_validation'], norm, xScaler),
                        'y_validation': utils.normalizeData(CVO['y_validation'], norm, yScaler),
                        'x_test': utils.normalizeData(CVO['x_test'], norm, xScaler),
                        'y_test': utils.normalizeData(CVO['y_test'], norm, yScaler)
                        })

    # Creation MLP model:self
    model = RegressionMLP(asset, savePath, dev)

    # training parameters
    nNeurons = range(1, CVA[-1]['x_train'].shape[1] + 1)

    # Debug
    # resultsObj = trainWrapper(15, model = model, CVA = CVAnorm, nInits = inits, epochs = 2000, validationSplit = 0.15,
    #                              loss = loss, optimizerAlgorithm = optimizer, verbose = verbose, dev = dev)
    # return

    # Start Parallel processing
    initTime = datetime.now()
    num_processes = multiprocessing.cpu_count()
    func = partial(trainWrapper, model = model, CVA = CVAnorm, nInits = inits, epochs = 2000, validationSplit = 0.15,
                                 loss = loss, optimizerAlgorithm = optimizer, verbose = verbose, dev = dev)
    p = multiprocessing.Pool(processes=num_processes)
    results = p.map(func, nNeurons)
    p.close()
    p.join()

    joblib.dump(results, '{}/Variables/{}_resultsWithCV{}.pkl'.format(savePath, asset, '_dev' if dev else ''))

    endTime = datetime.now()

    errorMean = []
    errorStd = []
    for result in results:
        errors = []
        for fold in result:
            errors.append(np.sqrt(fold['mse']))
        errorMean.append(np.array(errors).mean())
        errorStd.append(np.array(errors).std())
    minError = min(errorMean)
    minErrorNumberOfNeurons = errorMean.index(minError) + 1
    minErrorStd = errorStd[errorMean.index(minError)]

    fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
    ax.set_title('RMSE per number of neurons in hidden layer')
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('RMSE')
    ax.grid()
    ax.errorbar(range(1, len(errorMean) + 1), errorMean, yerr = errorStd, fmt = 'o')
    plt.figtext(0.5,  0.010, 'Lowest Validation RMSE {:.5f}+-{:.5f}, for {} neurons in the hidden layer'.format(minError, minErrorStd, minErrorNumberOfNeurons), size = 18, horizontalalignment = 'center')
    fig.savefig('{}/Figures/{}_RMSEperNumberOfNeurons{}.pdf'.format(savePath, asset, '_dev' if dev else ''), bbox_inches='tight')
    fig.savefig('{}/Figures/{}_RMSEperNumberOfNeurons{}.png'.format(savePath, asset, '_dev' if dev else ''), bbox_inches='tight')
    plt.close(fig)

    if (msg):
        t = endTime - initTime
        tStr = '{:02d}:{:02d}:{:02d}'.format(t.seconds//3600,(t.seconds//60)%60, t.seconds%60)
        message1 = u'{} Regression MLP ({}) Cross Validation analysis finished. Time elapsed: {}'.format(asset, norm, tStr)
        message2 = u'The best model had {} neurons in the hidden layer, with a mean error of {:.5f}+-{:.5f}.'.format(minErrorNumberOfNeurons, minError, minErrorStd)
        imgName = '{}/Figures/{}_RMSEperNumberOfNeurons{}'.format(savePath, asset, '_dev' if dev else '')
        try:
            bot.sendMessage([message1, message2], imgPath = imgName + '.png', filePath = imgName + '.pdf')
        except Exception:
            print('Falha ao enviar mensagem de conclusão')


if __name__ == "__main__":
    main()
