# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..') #src directory
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralstocks.utils import *
from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
# </editor-fold>

class MLP:
    def __init__(self, asset, savePath, **kwargs):
        self.analysis_str = 'MLP'
        self.dev          = kwargs['dev'] if 'dev' in kwargs else False
        self.savePath     = savePath
        self.saveFigPath  = savePath + '/Figures'
        self.saveVarPath  = savePath + '/Variables'
        self.saveModPath  = savePath + '/Models'
        self.asset        = asset
        self.bot          = kwargs['bot'] if 'bot' in kwargs else None

    def train(self, X, y, neuronsInHiddenLayer, **kwargs):
        '''
            Trains a MLP

            Parameters
            ----------
            X: input data of the ANN

            y: expected output of the ANN

            neuronsInHiddenLayer: Number of neurons in the ANN hidden layers

            norm: Normalisation technique used. Possible values are 'mapminmax', 'mapstd' and 'mapstd_rob'. Default 'mapminmax'

            nInits: number of initializations of the MLP. Default 1

            epochs: maximum number of epochs of training. Default 2000

            validationSplit: size of train/validation split. Default 0.15

            loss: loss function to use in training. Default SGD

            hiddenActivation: hidden layer activation function. Default tanh

            outputActivation: output layer activation function. Default linear

            patience: earlyStopping algorithm patience. Default 25

            Returns
            ----------

            """
        '''
        if self.dev: X = X[-400:]
        if self.dev: y = y[-400:]
        norm = kwargs['norm'] if ('norm' in kwargs) else 'mapminmax'
        xNorm, xScaler = normalizeData(x, norm)
        yNorm, yScaler = normalizeData(y, norm)
        nInits = kwargs['nInits'] if ('nInits' in kwargs and not self.dev) else 1
        inputDim = xNorm.shape[1]
        earlyStopping = EarlyStopping(monitor='val_loss', patience= kwargs['patience'] if 'patience' in kwars else 25, mode='auto')
        optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else 'SGD'
        loss = kwargs['loss'] if 'loss' in kwargs else 'mean_squared_error'
        hiddenActivation = kwargs['hiddenActivation'] if 'hidddataPathenActivation' in kwargs else 'tanh'
        outputActivation = kwargs['outputActivation'] if 'outputActivation' in kwargs else 'linear'

        bestValLoss = np.Inf
        bestFitHistory = None
        fitData = {}

        modelCheckpoint = ModelCheckpoint('{}.h5'.format(getSaveString(saveModPath, inputDim, neuronsInHiddenLayer, optimizer, norm)),
                                          save_best_only=True)
        for init in range(nInits):
            m_time = time.time()
            model = Sequential([Dense(neuronsInHiddenLayer, activation = hiddenActivation, input_dim = inputDim),
                                Dense(1, activation = outputActivation)
                               ])
            model.compile(optimizer = optimizer, loss = loss)

            fitHistory = model.fit(xNorm,
                                   yNorm,
                                   epochs = kwargs['epochs'] if 'epochs' in kwargs else 2000,
                                   verbose = 0,
                                   shuffle = True,
                                   validation_split = kwargs['validationSplit'] if 'validationSplit' in kwargs else 0.15,
                                   callbacks = [modelCheckpoint,
                                                earlyStopping])

            if min(fitHistory.history['val_loss']) < bestValLoss:
                bestFitHistory = fitHistory
                bestValLoss = min(fitHistory.history['val_loss'])

        joblib.dump(bestFitHistory, utils.getSaveString(saveVarPath, inputDim, neuronsInHiddenLayer, optimizer, norm, extra = 'fitHistory'))
        return bestValLoss

def trainRegressionMLP(neuronsInHiddenLayer, asset, savePath, X, y, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                       loss = 'mse', optimizerAlgorithm = 'sgd', hiddenActivation = 'tanh', outputActivation = 'linear', patience = 25,
                       verbose = False, dev = False):

    analysisStr = 'regression_MLP'

    saveFigPath = savePath + '/Figures'
    saveVarPath = savePath + '/Variables'
    saveModPath = savePath + '/Models'
    if not os.path.exists(saveVarPath): os.makedirs(saveVarPath)
    if not os.path.exists(saveFigPath): os.makedirs(saveFigPath)
    if not os.path.exists(saveModPath): os.makedirs(saveModPath)

    nInits = nInits if not dev else 1
    X = X if not dev else X[-400:]
    y = y if not dev else y[-400:]
    inputDim = X.shape[1]
    if (optimizerAlgorithm.upper() == 'SGD'): optimizer = optimizers.SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
    elif (optimizerAlgorithm.upper() == 'ADAM'): optimizer = optimizers.Adam(lr=0.0001)
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = patience, mode='auto')
    modelCheckpoint = ModelCheckpoint('{}.h5'.format(getSaveString(saveModPath, asset, analysisStr, inputDim, neuronsInHiddenLayer, optimizerAlgorithm, norm, dev = dev)),
                                      save_best_only=True)

    bestValLoss = np.Inf
    bestFitHistory = None

    initTime = time.time()
    for init in range(nInits):
        model = None # garantees model reset
        iTime = time.time()
        if verbose: print('Starting {} training ({:02d} neurons, init {})'.format(asset, neuronsInHiddenLayer, init))
        model = Sequential([Dense(neuronsInHiddenLayer, activation = hiddenActivation, input_dim = inputDim),
                            Dense(1, activation = outputActivation)
                           ])
        model.compile(optimizer = optimizer, loss = loss)

        fitHistory = model.fit(X,
                               y,
                               epochs = epochs,
                               verbose = 0,
                               shuffle = True,
                               validation_split = validationSplit,
                               callbacks = [modelCheckpoint,
                                            earlyStopping])

        if min(fitHistory.history['val_loss']) < bestValLoss:
            bestValLoss = min(fitHistory.history['val_loss'])
            bestFitHistory = fitHistory.history

        eTime = time.time()
        if verbose: print('Finished {} training ({:02d} neurons, init {}) -> Ellapsed time: {:.3f} seconds'.format(asset, neuronsInHiddenLayer, init, eTime - iTime))
    #end for nInits
    endTime = time.time()
    #if verbose: print('[-]Finished all {} trainings ({:02d} neurons) -> Ellapsed time: {:.3f} seconds'.format(asset, neuronsInHiddenLayer, endTime - initTime))

    joblib.dump(bestFitHistory, getSaveString(saveVarPath, asset, analysisStr, inputDim, neuronsInHiddenLayer, optimizerAlgorithm, norm, extra = 'fitHistory', dev = dev) + '.pkl')

    fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
    ax.set_title('RMSE per epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.grid()
    trainingSet, = ax.plot(np.sqrt(bestFitHistory['loss']), 'b', label = 'Training set')
    validationSet, = ax.plot(np.sqrt(bestFitHistory['val_loss']), 'r', label = 'Validation set')
    plt.legend(handles=[trainingSet, validationSet], labels=['Training set', 'Validation set'], prop={'size': 18})
    plt.figtext(0.5,  0.010, 'Lowest Validation RMSE: {:.5f}'.format(np.sqrt(min(bestFitHistory['val_loss']))), size = 18, horizontalalignment = 'center')
    fig.savefig(getSaveString(saveFigPath, asset, analysisStr, inputDim, neuronsInHiddenLayer, optimizerAlgorithm, norm, extra = 'fitHistory', dev = dev) + '.pdf', bbox_inches='tight')
    fig.savefig(getSaveString(saveFigPath, asset, analysisStr, inputDim, neuronsInHiddenLayer, optimizerAlgorithm, norm, extra = 'fitHistory', dev = dev) + '.png', bbox_inches='tight')

    return bestValLoss
