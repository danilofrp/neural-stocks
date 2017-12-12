import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import pandas as pd
from pyTaLib.indicators import *
from neuralstocks.preprocessing import *
from neuralstocks.plots import *
from neuralstocks.utils import *
from pyTaLib.indicators import *

def acquireData(filePath, replicateForHolidays = False, meanStdLen = None, returnCalcParams = [], SMAcols = [], SMAparams = [], EMAcols = [], EMAparams = [], dropNan = False):
    """
    Data Acquisition

    Parameters
    ----------
    filePath: string, location of the raw data file

    replicateForHolidays : bool, indicates wheter or not to insert holidays in the
        series, replicating the last value. Every missing day that is not weekend
        is considered to be a holiday. Default False

    meanStdLen: int, length of rolling mean and standart deviation calculation.
        Default None

    returnCalcParams : array-like, array containing one or two-sized string arrays,
        refering to the columns used to calculate returns. If the array contains a
        single column, calculates log returns between actual and previos samples.
        If the array specifies 2 columns, calculates log return between corresponding
        samples of both columns. Default empty

    SMAparams, EMAparams : tuple (string, int) array-like, set of desired columns to
        calculate moving averages from, together with specific MA length. Default empty

    dropNan: bool, indicates either or not to exclude any lines containing nan-values
        of the dataSet. Default False

    Returns
    ----------
    df : pandas.DataFrame, DataFrame containing original data and any aditional
        calculations specified in function params
    """
    df = pd.read_csv(filePath, delimiter=';', decimal=',',
                     parse_dates=['Date'], dayfirst=True, index_col='Date')
    df = df.sort_index() #csv entries begin from most recent to older dates

    if replicateForHolidays:
        df = insertHolidays(df)

    if meanStdLen:
        df = pd.concat([df, pd.Series(df['Close'].rolling(window=meanStdLen,center=False).mean(), name = 'Close_rollMean{}'.format(meanStdLen))], axis=1)
        df = pd.concat([df, pd.Series(df['Close'].rolling(window=meanStdLen,center=False).std(), name = 'Close_rollStd{}'.format(meanStdLen))], axis=1)

    for cols in returnCalcParams:
        if len(cols) == 1:
            df = pd.concat([df, logReturns(df[cols[0]])], axis=1)
        elif len(cols) == 2:
            df = pd.concat([df, logReturns(df[cols[0]], df[cols[1]])], axis=1)

    if len(SMAparams) > 0:
        for param in SMAparams:
            df = calculateSMAs(df, param[0], param[1])

    if len(EMAparams) > 0:
        for param in EMAparams:
            df = calculateEMAs(df, param[0], param[1])

    return df.dropna() if dropNan else df
