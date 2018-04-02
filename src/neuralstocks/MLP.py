# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..') #src directory
import abc
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

class Model:
    __metaclass__ = abc.ABCMeta
    inputDim = None
    hiddenLayers = []
    outputDim = 1
    norm = ''
    optimizerAlgorithm = ''
    loss = ''
    metrics = []
    patience = 0

    def __init__(self, asset, savePath, dev = False):
        self.analysisStr = self.__class__.__name__
        self.dev          = dev
        self.asset        = asset
        self.savePath     = savePath
        self.saveFigPath = savePath + '/Figures'
        self.saveVarPath = savePath + '/Variables'
        self.saveModPath = savePath + '/Models'
        createPath(self.saveVarPath)
        createPath(self.saveFigPath)
        createPath(self.saveModPath)

    def setTrainParams(self, inputDim, hiddenLayers, outputDim = 1, norm = 'mapminmax', optimizerAlgorithm = 'SGD', loss = 'mse', metrics = ['mae', 'acc'], patience = 25):
        self.inputDim = inputDim
        self.hiddenLayers = hiddenLayers
        self.outputDim = outputDim
        self.norm = norm
        self.optimizerAlgorithm = optimizerAlgorithm
        self.loss = loss
        self.metrics = metrics
        self.patience = patience

    def getSaveString(self, savePath, fold = None, extra = None):
        return '{}/{}_{}_{}_{}_{}{}{}{}'.format(savePath, self.asset, self.analysisStr, self.getNeuronsString(), self.optimizerAlgorithm, self.norm,
                                                '_' + fold if (fold is not None and fold is not '') else '', '_' + extra if (extra is not None and extra is not '') else '',
                                                '_dev' if self.dev else '')

    def getNeuronsString(self):
        neuronsString = str(self.inputDim) + 'x'
        if (isinstance(self.hiddenLayers, list)):
            for neurons in self.hiddenLayers:
                neuronsString += (str(neurons) + 'x')
        elif (isinstance(self.hiddenLayers, int)):
            neuronsString = str(self.hiddenLayers) + 'x'
        neuronsString += str(self.outputDim)
        return neuronsString

    @abc.abstractmethod
    def train(self, X, y, hiddenLayers, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                    hiddenActivation = 'tanh', outputActivation = 'linear', loss = 'mse', optimizerAlgorithm = 'sgd',
                    metrics = ['mae', 'acc'], patience = 25, verbose = False, dev = False):
        '''
            Trains the model

            Parameters
            ----------
            X: input data of the ANN

            y: expected output of the ANN

            hiddenLayers: list, Number of neurons in the ANN hidden layers

            norm: Normalisation technique used. Possible values are 'mapminmax', 'mapstd' and 'mapstd_rob'. Default 'mapminmax'

            nInits: int, number of initializations of the MLP. Default 1

            epochs: int, maximum number of epochs of training. Default 2000

            validationSplit: float, size of train/validation split. Default 0.15

            hiddenActivation: string, hidden layer activation function. Default tanh

            outputActivation: string, output layer activation function. Default linear

            loss: string, loss function to use in training. Default MSE

            optimizerAlgorithm: string, optimizer algorithm to be used in training. Default SGD

            metrics = string list, aditional metrics to evaluate over training. Default ['mae', 'acc']

            patience: int, earlyStopping algorithm patience. Default 25 epochs

            verbose: boolean, default False

            dev: boolean, default False

            Returns
            ----------

            bestValLoss: float, best validation loss achieved
        '''
        return

class RegressionMLP(Model):
    def __init__(self, asset, savePath, dev = False):
        Model.__init__(self, asset, savePath, dev)

    def train(self, X, y, hiddenLayers, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                    loss = 'mse', optimizerAlgorithm = 'sgd', hiddenActivation = 'tanh', outputActivation = 'linear',
                    metrics = ['mae', 'acc'], patience = 25, verbose = False, dev = False):
        self.setTrainParams(X.shape[1], hiddenLayers, y.shape[1], norm, optimizerAlgorithm, loss, metrics, patience)
        nInits = nInits if not self.dev else 1
        X = X if not self.dev else X[-400:]
        y = y if not self.dev else y[-400:]
        if (self.optimizerAlgorithm.upper() == 'SGD'): optimizer = optimizers.SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
        elif (self.optimizerAlgorithm.upper() == 'ADAM'): optimizer = optimizers.Adam(lr=0.0001)
        earlyStopping = EarlyStopping(monitor = 'val_loss', patience = patience, mode='auto')
        modelCheckpoint = ModelCheckpoint('{}.h5'.format(self.getSaveString(self.saveModPath)), save_best_only=True)

        bestValLoss = np.Inf
        bestFitHistory = None

        initTime = time.time()
        for init in range(nInits):
            model = None # garantees model reset
            iTime = time.time()
            if verbose: print('Starting {} training ({:02d} neurons, init {})'.format(self.asset, self.hiddenLayers, init))
            model = Sequential([Dense(self.hiddenLayers, activation = hiddenActivation, input_dim = self.inputDim),
                                Dense(1, activation = outputActivation)
                               ])
            model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)

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
            if verbose: print('Finished {} training ({:02d} neurons, init {}) -> Ellapsed time: {:.3f} seconds'.format(self.asset, self.hiddenLayers, init, eTime - iTime))
        #end for nInits
        endTime = time.time()

        joblib.dump(bestFitHistory, '{}.pkl'.format(self.getSaveString(self.saveVarPath, extra = 'fitHistory')))

        fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
        ax.set_title('RMSE per epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.grid()
        trainingSet, = ax.plot(np.sqrt(bestFitHistory['loss']), 'b', label = 'Training set')
        validationSet, = ax.plot(np.sqrt(bestFitHistory['val_loss']), 'r', label = 'Validation set')
        plt.legend(handles=[trainingSet, validationSet], labels=['Training set', 'Validation set'], prop={'size': 18})
        plt.figtext(0.5,  0.010, 'Lowest Validation RMSE: {:.5f}'.format(np.sqrt(min(bestFitHistory['val_loss']))), size = 18, horizontalalignment = 'center')
        fig.savefig('{}.pdf'.format(self.getSaveString(self.saveFigPath, extra = 'fitHistory')), bbox_inches='tight')
        fig.savefig('{}.png'.format(self.getSaveString(self.saveFigPath, extra = 'fitHistory')), bbox_inches='tight')

        return bestValLoss

def trainRegressionMLP(neuronsInHiddenLayer, X, y, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                       loss = 'mse', optimizerAlgorithm = 'sgd', hiddenActivation = 'tanh', outputActivation = 'linear',
                       patience = 25, verbose = False, dev = False):

    analysisStr = 'regression_MLP'

    saveFigPath = savePath + '/Figures'
    saveVarPath = savePath + '/Variables'
    saveModPath = savePath + '/Models'
    createPath(saveVarPath)
    createPath(saveFigPath)
    createPath(saveModPath)

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
