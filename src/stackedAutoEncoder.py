import sys, os
sys.path.append('./src')
import pickle
import numpy as np
import pandas as pd
import time

from keras.utils import np_utils
from keras import backend as K

import sklearn.metrics
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from neuralstocks import trainparameters as trnparams
from neuralstocks.SAE import StackedAutoEncoder
from neuralstocks.dataacquisition import acquireData
from neuralstocks.utils import prepData

import multiprocessing

init_time = time.time()

m_time = time.time()
print 'Time to import all libraries: '+str(m_time-init_time)+' seconds'

analysis_name = 'StackedAutoEncoder'

# Enviroment variables
#data_path = os.getenv('OUTPUTDATAPATH')
#results_path = os.getenv('PACKAGE_NAME')
asset = 'PETR4'
data_path = '../data/stocks/{}/diario/{}.CSV'.format(asset, asset)
IBOV_path = '../data/indexes/{}/diario/{}.CSV'.format('IBOV', 'IBOV')
USD_path = '../data/forex/{}/diario/{}.CSV'.format('USDBRL', 'USDBRL')
results_path = '../results/{}'.format(asset)

# paths to export results
base_results_path = '%s/%s'%(results_path,analysis_name)
pict_results_path = '%s/pictures_files'%(base_results_path)
files_results_path = '%s/output_files'%(base_results_path)

development_flag = False
development_events = 400

# For multiprocessing purpose
num_processes = multiprocessing.cpu_count()

# Read data
m_time = time.time()

PETR4 = acquireData(data_path)
IBOV = acquireData(IBOV_path)
USDBRL = acquireData(USD_path)
df = pd.concat([PETR4, IBOV, USDBRL], axis = 1).dropna()
columnsToUse = ['PETR4_Close_resid',
                'PETR4_Close_rollStd20',
                'PETR4_Close_returns', 'PETR4_Close/Open_returns', 'PETR4_High/Close_returns', 'PETR4_Low/Close_returns',
                'PETR4_Close_EMA17_logdiff', 'PETR4_Close_EMA72_logdiff', 'PETR4_Close_EMA200_logdiff', 'PETR4_Volume_EMA21_logdiff',
                'PETR4_MACD_12_26_9', 'PETR4_MACDsignal_12_26_9', 'PETR4_Bollinger%b_20', 'PETR4_OBV',
                'PETR4_Holiday',
                'IBOV_Close_rollStd20',
                'IBOV_Close_returns', 'IBOV_Close/Open_returns', 'IBOV_High/Close_returns', 'IBOV_Low/Close_returns',
                'IBOV_Close_EMA17_logdiff', 'IBOV_Close_EMA72_logdiff', 'IBOV_Close_EMA200_logdiff',
                'USDBRL_Close_rollStd20',
                'USDBRL_Close_returns', 'USDBRL_Close/Open_returns', 'USDBRL_High/Close_returns', 'USDBRL_Low/Close_returns',
                'USDBRL_Close_EMA17_logdiff', 'USDBRL_Close_EMA72_logdiff', 'USDBRL_Close_EMA200_logdiff',
               ]
# Process data
xTrain, yTrain, xTest, yTest = prepData(df = df,
                                        columnsToUse = columnsToUse, columnToPredict = 'PETR4_Close_resid',
                                        nDelays = 10, testSetSize = len(df['2017'])
                                       )
xScaler = MinMaxScaler(feature_range=(-1, 1))
yScaler = MinMaxScaler(feature_range=(-1, 1))
xScaler.fit(xTrain)
yScaler.fit(yTrain)
xTrainScaled = xScaler.transform(xTrain)
xTestScaled = xScaler.transform(xTest)
yTrainScaled = yScaler.transform(yTrain)

# Load train parameters
analysis_str = 'StackedAutoEncoder'
model_prefix_str = 'PETR4'

trn_params_folder='%s/%s/%s_trnparams.jbl'%(results_path,analysis_str,analysis_name)
if os.path.exists(trn_params_folder):
    os.remove(trn_params_folder)
if not os.path.exists(trn_params_folder):
    trn_params = trnparams.NeuralRegressionTrnParams(n_inits=1,
                                                     hidden_activation='tanh', # others tanh, relu, sigmoid, linear
                                                     output_activation='tanh',
                                                     n_epochs=500,  #500
                                                     patience=30,  #30
                                                     batch_size=1024, #128
                                                     verbose=False,
                                                     optmizerAlgorithm='Adam',
                                                     metrics=['accuracy'],
                                                     loss='mse')
    trn_params.save(trn_params_folder)
else:
    trn_params = trnparams.NeuralClassificationTrnParams()
    trn_params.load(trn_params_folder)

print trn_params.get_params_str()

if development_flag:
    print '[+] Development mode'

# Create neurons vector to be used in multiprocessing.Pool()
SAE = StackedAutoEncoder(params           = trn_params,
                         development_flag = development_flag,
                         save_path        = results_path,
                         prefix_str       = model_prefix_str,
                         regression       = True)

# Choose neurons topology
hidden_neurons = [20, 10, 10, 5]
print hidden_neurons

regularizer = "" #dropout / l1 / l2
regularizer_param = 0.3

# Choose layer to be trained
layer = 4
# Functions defined to be used by multiprocessing.Pool()
def trainNeuron(ineuron):
    SAE.trainLayer(data = xTrain,
                   trgt = yTrain,
                   hidden_neurons = hidden_neurons[:layer],# + [ineuron],
                   layer = layer,
                   regularizer = regularizer,
                   regularizer_param = regularizer_param)


start_time = time.time()

if K.backend() == 'theano':
    # Start Parallel processing
    p = multiprocessing.Pool(processes=num_processes)

    ####################### SAE LAYERS ############################
    # It is necessary to choose the layer to be trained

    # To train on multiple cores sweeping the number of folds
    # folds = range(len(CVO[inovelty]))
    # results = p.map(trainFold, folds)

    # To train multiple topologies sweeping the number of neurons
    neurons_mat = [10, 20] + range(50,450,50)
    # results = p.map(trainNeuron, neurons_mat)

    p.close()
    p.join()
else:
    neurons_mat = [10, 20] + range(50,450,50)
    # for ifold in range(len(CVO[inovelty])):
    #     result = trainFold(ifold)
    for ineuron in neurons_mat[:len(neurons_mat)-layer+2]:
        print '[*] Training Layer %i - %i Neurons'%(layer, ineuron)
        result = trainNeuron(ineuron)

end_time = time.time() - start_time
print "It took %.3f seconds to perform the training"%(end_time)


model = SAE.getEncoder(data = xTrain,
                       trgt = yTrain,
                       hidden_neurons = hidden_neurons[:layer],# + [ineuron],
                       layer = layer,
                       regularizer = regularizer,
                       regularizer_param = regularizer_param)
for l in range(len(model.layers)):
    model.layers[l].trainable = False
model.summary()

from keras.layers import Dense

model.add(Dense(1, activation='linear'))
model.compile(optimizer='Adam', loss = 'mse')
model.summary()
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(xTrain, yTrain, epochs=500, callbacks=[earlyStopping], validation_split=0.15)

from neuralstocks.plots import plotSeries
%matplotlib inline

predictions = yScaler.inverse_transform(model.predict(xTestScaled))
predictions = np.reshape(predictions, -1)
predictionsSeries = pd.Series(predictions, index = df['2017'].index, name='SAE_Predictions')
PETR4_predicted = pd.Series(df['PETR4_Close_trend']['2017'] + predictionsSeries,  name = 'PETR4_predicted')

plotSeries([df['PETR4_Close'],
#            df['PETR4_Close_trend'],
            PETR4_predicted],
           initialPlotDate = '2017-05', finalPlotDate = '2017-06',
           title = 'Original Data vs Predicted', ylabel = 'Price')

plotSeries([df['PETR4_Close'] - df['PETR4_Close_trend'],
            df['PETR4_Close'] - (df['PETR4_Close_trend'] + predictionsSeries)],
           initialPlotDate = '2017-05', finalPlotDate = '2017-06',
           title = 'Original Data vs Predicted', ylabel = 'Error')


plotSeries([predictionsSeries],
           initialPlotDate = '2017-04', finalPlotDate = '2017-08',
           title = 'Predicted residuals', ylabel = 'Error')
