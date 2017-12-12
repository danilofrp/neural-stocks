import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import numpy as np
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf

def getWeights(s, window, weightModel, weightModelWindow):
    if weightModel == 'full_pgram':
        weights = list(reversed(periodogram(s.dropna())[1][1 : window + 1]))
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif weightModel == 'full_acorr':
        weights = list(reversed(np.abs(acf(s.dropna(), nlags= window + 1))[1 : window + 1]))
        weightModelWindow = weightModelWindow if weightModelWindow else window
    elif weightModel == 'window_pgram':
        weightModelWindow = weightModelWindow if weightModelWindow or (not weightModelWindow >= 2 * window) else 2 * window
        weights = list(reversed(periodogram(s.dropna())[1][1 : window + 1]))
        if not checkIfTwoOrMoreValuesAreNotZero(weights):
            weights = None
    elif weightModel == 'window_acorr':
        weights = list(reversed(np.abs(acf(s.dropna(), nlags= window + 1))[1 : window + 1]))
        if np.isnan(weights).all() or not checkIfTwoOrMoreValuesAreNotZero(weights):
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
    a = np.polyfit(x, y, deg = fitOrder, w = weights);
    prediction = 0
    for j in range(fitOrder, -1, -1):
        prediction += a[fitOrder - j]*(window**j)
    return prediction

def crosscorrelation(x, y, nlags = 0):
    """Cross correlations calculatins until nlags.
    Parameters
    ----------
    x, y : pandas.Series objects of equal length
    nlags : int, number of lags to calculate cross-correlation, default 0

    Returns
    ----------
    crosscorrelation : [float]
    """
    return [x.corr(y.shift(lag)) for lag in range(nlags + 1)]

def KLDiv(p, q, nBins, bins = np.array([-1,0, 1])):
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
