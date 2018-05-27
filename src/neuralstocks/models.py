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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
# </editor-fold>

class BaseModel:
    __metaclass__ = abc.ABCMeta
    inputDim = None
    hiddenLayers = []
    outputDim = 1
    norm = ''
    optimizerAlgorithm = ''
    loss = ''
    metrics = []
    patience = 0
    xScaler = None
    model = None

    def __init__(self, asset, savePath, verbose = False, dev = False):
        self.analysisStr = self.__class__.__name__
        self.dev         = dev
        self.verbose     = verbose
        self.asset       = asset
        self.savePath    = savePath
        self.saveFigPath = savePath + '/Figures'
        self.saveVarPath = savePath + '/Variables'
        self.saveModPath = savePath + '/Models'
        createPath(self.saveVarPath)
        createPath(self.saveFigPath)
        createPath(self.saveModPath)

    def setTrainParams(self, inputDim, hiddenLayers, outputDim = 1, norm = 'mapminmax', optimizerAlgorithm = 'SGD',
                       hiddenActivation = 'tanh', outputActivation = 'linear', loss = 'mse', metrics = ['mae'],
                       validationSplit = 0.15, epochs = 2000, patience = 25, verbose = False, dev = False):
        if inputDim:             self.inputDim = inputDim
        if hiddenLayers:         self.hiddenLayers = hiddenLayers
        if outputDim:            self.outputDim = outputDim
        if norm:                 self.norm = norm
        if optimizerAlgorithm:   self.optimizerAlgorithm = optimizerAlgorithm
        if hiddenActivation:     self.hiddenActivation = hiddenActivation
        if outputActivation:     self.outputActivation = outputActivation
        if loss:                 self.loss = loss
        if metrics:              self.metrics = metrics
        if validationSplit:      self.validationSplit = validationSplit
        if epochs:               self.epochs = epochs
        if patience:             self.patience = patience
        if verbose != self.verbose:              self.verbose = verbose
        if dev != self.dev:                  self.dev = dev

    def getSaveString(self, savePath, neuronsString = None, fold = None, extra = None):
        neuronsString = neuronsString if neuronsString else self.getNeuronsString()
        return '{}/{}_{}_{}_{}_{}{}{}{}'.format(savePath, self.asset, self.analysisStr, neuronsString, self.optimizerAlgorithm, self.norm,
                                                '_fold' + str(fold) if (fold is not None and fold is not '') else '', '_' + extra if (extra is not None and extra is not '') else '',
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

    def normalizeInputData(data, norm, force = False):
        '''
            Method that preprocess data normalizing it according to 'norm' parameter.
        '''
        #normalize data based in train set
        if (not self.xScaler) or force:
            if norm == 'mapstd':
                self.xScaler = StandardScaler().fit(data)
            elif norm == 'mapstd_rob':
                self.xScaler = RobustScaler().fit(data)
            elif norm == 'mapminmax':
                self.xScaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        norm_data = self.xScaler.transform(data)

        return norm_data, xScaler

    def normalizeOutputData(data, norm, force = False):
        '''
            Method that preprocess data normalizing it according to 'norm' parameter.
        '''
        #normalize data based in train set
        if (not self.yScaler) or force:
            if norm == 'mapstd':
                self.yScaler = StandardScaler().fit(data)
            elif norm == 'mapstd_rob':
                self.yScaler = RobustScaler().fit(data)
            elif norm == 'mapminmax':
                self.yScaler = MinMayScaler(feature_range=(-1, 1)).fit(data)
        norm_data = self.yScaler.transform(data)

        return norm_data, yScaler

    def denormalizeOutputData(data):
        return self.yScaler.inverse_transform(data)

    @abc.abstractmethod
    def train(self, X, y, hiddenLayers, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                    hiddenActivation = 'tanh', outputActivation = 'linear', loss = 'mse', optimizerAlgorithm = 'sgd',
                    metrics = ['mae'], patience = 25, verbose = False, dev = False):
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

            metrics = string list, aditional metrics to evaluate over training. Default ['mae']

            patience: int, earlyStopping algorithm patience. Default 25 epochs

            verbose: boolean, default False

            dev: boolean, default False

            Returns
            ----------

            bestValLoss: float, best validation loss achieved
        '''
        return

class RegressionMLP(BaseModel):
    def __init__(self, asset, savePath, verbose = False, dev = False):
        BaseModel.__init__(self, asset, savePath, verbose, dev)

    def train(self, X, y, hiddenLayers, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                    loss = 'mse', optimizerAlgorithm = 'sgd', hiddenActivation = 'tanh', outputActivation = 'linear',
                    metrics = ['mae'], patience = 25, verbose = False, dev = False):
        self.setTrainParams(X.shape[1], hiddenLayers, y.shape[1], norm, optimizerAlgorithm, hiddenActivation, outputActivation, loss, metrics, validationSplit, epochs, patience, verbose, dev)
        nInits = nInits if not self.dev else 1
        X = X if not self.dev else X[-400:]
        y = y if not self.dev else y[-400:]
        if (self.optimizerAlgorithm.upper() == 'SGD'): optimizer = SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
        elif (self.optimizerAlgorithm.upper() == 'ADAM'): optimizer = Adam(lr=0.0001)
        earlyStopping = EarlyStopping(monitor = 'val_loss', patience = patience, mode='auto')
        modelCheckpoint = ModelCheckpoint('{}.h5'.format(self.getSaveString(self.saveModPath)), save_best_only=True)

        bestValLoss = np.Inf
        bestFitHistory = None

        initTime = time.time()
        for init in range(nInits):
            model = None # garantees model reset
            iTime = time.time()
            if self.verbose: print('Starting {} training ({:02d} neurons, init {})'.format(self.asset, self.hiddenLayers, init))
            model = Sequential([Dense(self.hiddenLayers, activation = self.hiddenActivation, input_dim = self.inputDim),
                                Dense(1, activation = self.outputActivation)
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

    def trainWithCrossValidation(self, CVA, hiddenLayers, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
                                 loss = 'mse', optimizerAlgorithm = 'sgd', hiddenActivation = 'tanh', outputActivation = 'linear',
                                 metrics = ['mae'], patience = 25, verbose = False, dev = False):
        self.setTrainParams(CVA[0]['x_train'].shape[1], hiddenLayers, CVA[0]['y_train'].shape[1], norm, optimizerAlgorithm, hiddenActivation, outputActivation, loss, metrics, validationSplit, epochs, patience, verbose, dev)
        earlyStopping = EarlyStopping(monitor = 'val_loss', patience = patience, mode='auto')
        nInits = nInits if not self.dev else 1

        resultsArray = []
        fold = 0;
        for CVO in CVA:
            results = {}
            fold += 1
            CVO['x_train'] = CVO['x_train'] if not self.dev else CVO['x_train'][-400:]
            CVO['y_train'] = CVO['y_train'] if not self.dev else CVO['y_train'][-400:]
            modelCheckpoint = ModelCheckpoint('{}.h5'.format(self.getSaveString(self.saveModPath, fold = fold)), save_best_only=True)

            bestValLoss = np.Inf
            bestFitHistory = None

            initTime = time.time()
            for init in range(nInits):
                K.clear_session()
                iTime = time.time()
                if self.verbose: print('Starting {} training ({:02d} neurons, fold {}, init {})'.format(self.asset, self.hiddenLayers, fold, init))
                if (self.optimizerAlgorithm.upper() == 'SGD'): optimizer = SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
                elif (self.optimizerAlgorithm.upper() == 'ADAM'): optimizer = Adam(lr=0.0001)
                model = Sequential([Dense(self.hiddenLayers, activation = self.hiddenActivation, input_dim = self.inputDim),
                                    Dense(1, activation = self.outputActivation)
                                   ])
                model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)

                fitHistory = model.fit(CVO['x_train'],
                                       CVO['y_train'],
                                       epochs = epochs,
                                       verbose = 0,
                                       shuffle = False,
                                       validation_data = (CVO['x_validation'], CVO['y_validation']),
                                       callbacks = [modelCheckpoint,
                                                    earlyStopping])

                if min(fitHistory.history['val_loss']) < bestValLoss:
                    bestValLoss = min(fitHistory.history['val_loss'])
                    bestFitHistory = fitHistory.history

                eTime = time.time()
                if verbose: print('Finished {} training ({:02d} neurons, fold {}, init {}) -> Ellapsed time: {:.3f} seconds'.format(self.asset, self.hiddenLayers, fold, init, eTime - iTime))
            #end for nInits
            endTime = time.time()

            joblib.dump(bestFitHistory, '{}.pkl'.format(self.getSaveString(self.saveVarPath, fold = fold, extra = 'fitHistory')))

            fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
            ax.set_title('RMSE per epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RMSE')
            ax.grid()
            trainingSet, = ax.plot(np.sqrt(bestFitHistory['loss']), 'b', label = 'Training set')
            validationSet, = ax.plot(np.sqrt(bestFitHistory['val_loss']), 'r', label = 'Validation set')
            plt.legend(handles=[trainingSet, validationSet], labels=['Training set', 'Validation set'], prop={'size': 18})
            plt.figtext(0.5,  0.010, 'Lowest Validation RMSE: {:.5f}'.format(np.sqrt(min(bestFitHistory['val_loss']))), size = 18, horizontalalignment = 'center')
            fig.savefig('{}.pdf'.format(self.getSaveString(self.saveFigPath, fold = fold, extra = 'fitHistory')), bbox_inches='tight')
            fig.savefig('{}.png'.format(self.getSaveString(self.saveFigPath, fold = fold, extra = 'fitHistory')), bbox_inches='tight')
            plt.close(fig)

            model = load_model('{}.h5'.format(self.getSaveString(self.saveModPath, fold = fold)))
            scores = model.evaluate(CVO['x_validation'], CVO['y_validation'], verbose = 0)
            results['mse'] = scores[0]
            results['mae'] = scores[1]
            resultsArray.append(results)

        return resultsArray

class RegressionSAE(BaseModel):
    def __init__(self, asset, savePath, verbose = False, dev = False):
        BaseModel.__init__(self, asset, savePath, verbose, dev)

    def train(self, X, y, hiddenLayers, norm = 'mapminmax', nInits = 1, epochs = 2000, validationSplit = 0.15,
              loss = 'mse', optimizerAlgorithm = 'sgd', hiddenActivation = 'tanh', outputActivation = 'linear',
              metrics = ['mae'], patience = 25, verbose = False, force = False, dev = False):
        self.setTrainParams(X.shape[1], hiddenLayers, y.shape[1], norm, optimizerAlgorithm, hiddenActivation, outputActivation, loss, metrics, validationSplit, epochs, patience, verbose, dev)
        if (self.optimizerAlgorithm.upper() == 'SGD'): optimizer = SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
        elif (self.optimizerAlgorithm.upper() == 'ADAM'): optimizer = Adam(lr=0.0001)
        nInits = nInits if not self.dev else 1
        X = X if not self.dev else X[-400:]
        y = y if not self.dev else y[-400:]
        if self.verbose: print('Starting {} {} SAE ({}) training'.format(self.asset, self.getNeuronsString(), self.norm))
        self.model = Sequential()

        for i in range(len(hiddenLayers)):
            layerInputDim = X.shape[1] if i == 0 else hiddenLayers[i - 1]
            nNeurons = hiddenLayers[i]
            neuronsString = '{:02d}x{:02d}x{:02d}'.format(layerInputDim, nNeurons, layerInputDim)
            earlyStopping = EarlyStopping(monitor = 'val_loss', patience = patience, mode='auto')
            modelCheckpoint = ModelCheckpoint('{}.h5'.format(self.getSaveString(self.saveModPath)), save_best_only=True)

            if (force or not os.path.exists('{}.h5'.format(self.getSaveString(self.saveModPath, neuronsString = neuronsString)))):
                if self.verbose: print('Training SAE layer {} ({} autoencoder)'.format(i + 1, neuronsString))
                xSet = X if i == 0 else self.model.predict(X)
                self.trainLayer(xSet, nNeurons, nInits)
            else:
                if self.verbose: print ('Layer {} (autoencoder {}) was previously trained, loading existing model'.format(i + 1, neuronsString))

            autoencoder = load_model('{}.h5'.format(self.getSaveString(self.saveModPath, neuronsString = neuronsString)))
            encoderLayer = autoencoder.get_layer(index=1)
            encoderLayer.trainable = False
            self.model.add(encoderLayer)
            self.model.compile(optimizer = optimizer, loss=loss)

        self.model.add(Dense(1, activation = self.outputActivation, name = 'output'))
        self.model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)
        if self.verbose: print('Training {} SAE ({}) output layer'.format(self.asset, self.getNeuronsString()))
        iTime = time.time()
        fitHistory = self.model.fit(X,
                                    y,
                                    epochs = self.epochs,
                                    verbose = 0,
                                    shuffle = True,
                                    validation_split = self.validationSplit,
                                    callbacks = [modelCheckpoint,
                                                 earlyStopping])
        if self.verbose: print('Finished {} training ({} SAE) -> Ellapsed time: {:.3f} seconds'.format(self.asset, self.getNeuronsString(), time.time() - iTime))
        joblib.dump(fitHistory.history, '{}.pkl'.format(self.getSaveString(self.saveVarPath, extra = 'fitHistory')))

        fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
        ax.set_title('RMSE per epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.grid()
        trainingSet, = ax.plot(np.sqrt(fitHistory.history['loss']), 'b', label = 'Training set')
        validationSet, = ax.plot(np.sqrt(fitHistory.history['val_loss']), 'r', label = 'Validation set')
        plt.legend(handles=[trainingSet, validationSet], labels=['Training set', 'Validation set'], prop={'size': 18})
        plt.figtext(0.5,  0.010, 'Lowest Validation RMSE: {:.5f}'.format(np.sqrt(min(fitHistory.history['val_loss']))), size = 18, horizontalalignment = 'center')
        fig.savefig('{}.pdf'.format(self.getSaveString(self.saveFigPath, extra = 'fitHistory')), bbox_inches='tight')
        fig.savefig('{}.png'.format(self.getSaveString(self.saveFigPath, extra = 'fitHistory')), bbox_inches='tight')

        return min(np.sqrt(fitHistory.history['val_loss']))

    def trainLayer(self, X, nNeurons, nInits):
        if (self.optimizerAlgorithm.upper() == 'SGD'): optimizer = SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
        elif (self.optimizerAlgorithm.upper() == 'ADAM'): optimizer = Adam(lr=0.001)
        earlyStopping = EarlyStopping(monitor = 'val_loss', patience = self.patience, mode='auto')
        neuronsString = '{:02d}x{:02d}x{:02d}'.format(X.shape[1], nNeurons, X.shape[1])
        modelCheckpoint = ModelCheckpoint('{}.h5'.format(self.getSaveString(self.saveModPath, neuronsString = neuronsString)), save_best_only=True)

        bestValLoss = np.Inf
        bestFitHistory = None

        initTime = time.time()
        for init in range(1, nInits + 1):
            model = None # garantees model reset
            iTime = time.time()
            if self.verbose: print('Starting {} training ({} neurons, init {})'.format(self.asset, neuronsString, init))
            model = Sequential([Dense(nNeurons, activation = self.hiddenActivation, input_dim = X.shape[1], name = 'dense_{}x{}'.format(X.shape[1], nNeurons)),
                                Dense(X.shape[1], activation = self.hiddenActivation)
                               ])
            model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)

            fitHistory = model.fit(X,
                                   X,
                                   epochs = self.epochs,
                                   verbose = 0,
                                   shuffle = True,
                                   validation_split = self.validationSplit,
                                   callbacks = [modelCheckpoint,
                                                earlyStopping])

            if min(fitHistory.history['val_loss']) < bestValLoss:
                bestValLoss = min(fitHistory.history['val_loss'])
                bestFitHistory = fitHistory.history

            eTime = time.time()
            if self.verbose: print('Finished {} training ({} neurons, init {}) -> Ellapsed time: {:.3f} seconds'.format(self.asset, neuronsString, init, eTime - iTime))
        #end for nInits
        endTime = time.time()

        joblib.dump(bestFitHistory, '{}.pkl'.format(self.getSaveString(self.saveVarPath, neuronsString = neuronsString, extra = 'fitHistory')))

        fig, ax = plt.subplots(figsize = (10,10), nrows = 1, ncols = 1)
        ax.set_title('RMSE per epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.grid()
        trainingSet, = ax.plot(np.sqrt(bestFitHistory['loss']), 'b', label = 'Training set')
        validationSet, = ax.plot(np.sqrt(bestFitHistory['val_loss']), 'r', label = 'Validation set')
        plt.legend(handles=[trainingSet, validationSet], labels=['Training set', 'Validation set'], prop={'size': 18})
        plt.figtext(0.5,  0.010, 'Lowest Validation RMSE: {:.5f}'.format(np.sqrt(min(bestFitHistory['val_loss']))), size = 18, horizontalalignment = 'center')
        fig.savefig('{}.pdf'.format(self.getSaveString(self.saveFigPath, neuronsString = neuronsString, extra = 'fitHistory')), bbox_inches='tight')
        fig.savefig('{}.png'.format(self.getSaveString(self.saveFigPath, neuronsString = neuronsString, extra = 'fitHistory')), bbox_inches='tight')

        return bestValLoss


# <editor-fold> standalone train function

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
    if (optimizerAlgorithm.upper() == 'SGD'): optimizer = SGD(lr=0.001, momentum=0.00, decay=0.0, nesterov=False)
    elif (optimizerAlgorithm.upper() == 'ADAM'): optimizer = Adam(lr=0.0001)
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
# </editor-fold>
