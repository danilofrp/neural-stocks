from __future__ import print_function
import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
from pandas.core.nanops import nanmean as pd_nanmean
from scipy.signal import periodogram
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def getWeights(s, window, weightModel, weightModelWindow):
    if weightModel == 'full_pgram':
        weights = list(reversed(periodogram(s.dropna())[1][1 : window + 1]))
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif weightModel == 'full_acorr':
        weights = list(reversed(np.abs(autocorrelation(s, nlags = window))[1 : window + 1]))
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif weightModel == 'window_pgram':
        weightModelWindow = weightModelWindow if weightModelWindow and (weightModelWindow >= 2 * window) else 2 * window
        weights = list(reversed(periodogram(s.dropna())[1][1 : window + 1]))
        if not checkIfTwoOrMoreValuesAreNotZero(weights):
            weightModelWindow = window
            weights = None
    elif weightModel == 'window_acorr':
        weightModelWindow = weightModelWindow if weightModelWindow and (weightModelWindow >= 2 * window) else 2 * window
        weights = list(reversed(np.abs(autocorrelation(s, nlags = window))[1 : window + 1]))
        if np.isnan(weights).any() or not checkIfTwoOrMoreValuesAreNotZero(weights):
            weightModelWindow = window
            weights = None
    elif not weightModel:
        weightModelWindow = window
        weights = None
    return weights, weightModelWindow

def checkIfTwoOrMoreValuesAreNotZero(x):
    notZero = False
    for i in range(len(x)):
        if notZero and x[i] != 0:
            return True
        elif x[i] != 0:
            notZero = True
    return False

def predict(x, y, fitOrder, weights, window):
    a = np.polyfit(x, y, deg = fitOrder, w = weights)
    prediction = 0
    for j in range(fitOrder, -1, -1):
        prediction += a[fitOrder - j]*(window**j)
    return prediction

def seasonalMean(s, freq):
    return np.array([pd_nanmean(s[i::freq]) for i in range(freq)])

def crosscorrelation(x, y, nlags = 0):
    """Cross correlations calculatins until nlags.
    Parameters
    ----------
    x, y : pandas.Series objects of equal lenght
    nlags : int, number of lags to calculate cross-correlation, default 0

    Returns
    ----------
    crosscorrelation : [float]
    """
    return [x.corr(y.shift(lag)) for lag in range(nlags + 1)]

def autocorrelation(x, nlags = 0):
    """Autocorrelation calculatins until nlags.
    Parameters
    ----------
    x: pandas.Series object
    nlags : int, number of lags to calculate cross-correlation, default 0

    Returns
    ----------
    autocorrelation : [float]
    """
    return [x.corr(x.shift(lag)) for lag in range(nlags + 1)]

def KLDiv(p, q, nBins = 100, bins = np.array([-1,0, 1])):
    maximum = max(p.dropna().max(), q.dropna().max())
    minimum = min(p.dropna().min(), q.dropna().min())
    [p_pdf,p_bins] = np.histogram(p.dropna(), bins = nBins, range = (minimum, maximum), density = True)
    [q_pdf,q_bins] = np.histogram(q.dropna(), bins = nBins, range = (minimum, maximum), density = True)
    kl_values = []
    for i in range(len(p_pdf)):
        if p_pdf[i] == 0 or q_pdf[i] == 0 :
            kl_values = np.append(kl_values,0)
        else:
            kl_value = np.abs(p_pdf[i]*np.log10(p_pdf[i]/q_pdf[i]))
            if np.isnan(kl_value):
                kl_values = np.append(kl_values,0)
            else:
                kl_values = np.append(kl_values,kl_value)
    return np.sum(kl_values)

sign = lambda a: int(a>0) - int(a<0)

def prepData(df, columnsToUse, columnToPredict, nDelays, testSetSize, validationSplitSize = None):
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []
    for i in range(len(df)):
        xTrainAux = []
        yTrainAux = []
        xTestAux = []
        yTestAux = []
        if i >= nDelays and (i < (len(df) - testSetSize)):
            for column in columnsToUse:
                if column == columnToPredict:
                    xTrainAux.extend(df[column][i - nDelays : i])
                elif 'Holiday' in column:
                    if df[column][i : i + 1].empty:
                        xTrainAux.extend([0])
                    else:
                        xTrainAux.extend(df[column][i : i + 1])
                else:
                    xTrainAux.extend(df[column][i - 1 : i])
            yTrainAux.append(df[columnToPredict][i])
            xTrain.append(xTrainAux)
            yTrain.append(yTrainAux)
        elif i >= nDelays and (i >= (len(df) - testSetSize)):
            for column in columnsToUse:
                if column == columnToPredict:
                    xTestAux.extend(df[column][i - nDelays : i])
                elif 'Holiday' in column:
                    if df[column][i : i + 1].empty:
                        xTestAux.extend([0])
                    else:
                        xTestAux.extend(df[column][i : i + 1])
                else:
                    xTestAux.extend(df[column][i - 1 : i])
            yTestAux.append(df[columnToPredict][i])
            xTest.append(xTestAux)
            yTest.append(yTestAux)

    if not validationSplitSize:
        return np.array(xTrain), np.array(yTrain), np.array(xTest), np.array(yTest)
    else:
        xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size = validationSplitSize)
        return np.array(xTrain), np.array(yTrain), np.array(xVal), np.array(yVal), np.array(xTest), np.array(yTest)

def reconstructReturns(observed, predictedReturns):
    predicted = observed.shift() * np.exp(predictedReturns)
    return predicted

def calculateRMSE(s1, s2):
    if len(s1) != len(s2):
        print('Error: both series must have equal lenght')
    else:
        return np.sqrt(np.square(s1 - s2).sum()/len(s1))

def calculateMAE(s1, s2):
    if len(s1) != len(s2):
        print('Error: both series must have equal lenght')
    else:
        return (np.abs(s1 - s2).sum())/len(s1)

def setPaths(f):
    dataPath = os.path.dirname(os.path.abspath(f)).split('neural-stocks', 1)[0] + 'data'
    savePath = os.path.dirname(os.path.abspath(f)).replace('neural-stocks', 'ns-results')
    return dataPath, savePath

def normalizeData(data, norm, scaler = None):
    '''
        Method that preprocess data normalizing it according to 'norm' parameter.
    '''
    #normalize data based in train set
    returnScaler = False
    if not scaler:
        returnScaler = True
        if norm == 'mapstd':
            scaler = StandardScaler().fit(data)
        elif norm == 'mapstdrob':
            scaler = RobustScaler().fit(data)
        elif norm == 'mapminmax':
            scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
    norm_data = scaler.transform(data)

    if returnScaler:
        return norm_data, scaler
    else:
        return norm_data

def createPath(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError, e:
            if e.errno != os.errno.EEXIST:
                raise
            pass

def customTimeSeriesSplit(df, nSplits, testStart = 2017, testEnd = 2017):
    CVA = []
    for n in range(nSplits):
        CVA.append({'train': df[:str(testStart - (nSplits - n) - 1)], 'validation': df[str(testStart - (nSplits - n)):str(testStart - (nSplits - n))], 'test': df[str(testStart):str(testEnd)]})
    return CVA

def prepDataWithCrossValidation(df, columnsToUse, columnToPredict, nDelays, nSplits, testStart = 2017, testEnd = 2017):
    x = []
    y = []
    for i in range(len(df)):
        xAux = []
        yAux = []
        if i >= nDelays:
            for column in columnsToUse:
                if column == columnToPredict:
                    xAux.extend(df[column][i - nDelays : i])
                elif 'Holiday' in column:
                    if df[column][i : i + 1].empty:
                        xAux.extend([0])
                    else:
                        xAux.extend(df[column][i : i + 1])
                else:
                    xAux.extend(df[column][i - 1 : i])
            yAux.append(df[columnToPredict][i])
            x.append(xAux)
            y.append(yAux)

    testSize = len(df[str(testStart):str(testEnd)])
    CVA = []
    for n in range(nSplits):
        trainStartIndex = 0
        trainEndIndex = len(df[:str(testStart - (nSplits - n) - 1)]) - nDelays
        validationStartIndex = trainEndIndex
        validationEndIndex = len(df[:str(testStart - (nSplits - n))]) - nDelays
        CVA.append({
                    'x_train': np.array(x[trainStartIndex : trainEndIndex]),
                    'y_train': np.array(y[trainStartIndex : trainEndIndex]),
                    'x_validation': np.array(x[validationStartIndex : validationEndIndex]),
                    'y_validation': np.array(y[validationStartIndex : validationEndIndex]),
                    'x_test': np.array(x[-testSize:]),
                    'y_test': np.array(y[-testSize:])
                    })

    return CVA


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
