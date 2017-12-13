import sys, os
sys.path.append('/home/danilofrp/projeto_final/neural-stocks/src')
import pandas as pd
from pyTaLib.indicators import *
from neuralstocks.preprocessing import *
from neuralstocks.plots import *
from neuralstocks.utils import *
from pyTaLib.indicators import *

def acquireData(filePath, replicateForHolidays = False, meanStdLen = None, returnCalcParams = [], SMAparams = [], EMAparams = [],
                MACDParams = [], BBParams = [], OBVParams = [], deTrendParams = None, colPrefix = None, dropNan = False):
    """
    Data Acquisition

    Parameters
    ----------
    filePath: string, location of the raw data file

    replicateForHolidays : bool, indicates wheter or not to insert holidays in the
        series, replicating the last value. Every missing day that is not weekend
        is considered to be a holiday. Default False

    meanStdLen: int, lenght of rolling mean and standart deviation calculation.
        Default None

    returnCalcParams : array of array, array containing one or two-sized string arrays,
        refering to the columns used to calculate returns. If the array contains a
        single column, calculates log returns between actual and previos samples.
        If the array specifies 2 columns, calculates log return between corresponding
        samples of both columns. Default empty

    SMAparams, EMAparams : {'column': value, 'lenght': value} array.
        Set of desired columns to calculate moving averages from, together with specific
        MA lenght. Default empty

    MACDParams : {'fast_lenght': value, 'slow_lenght': value, 'signal_lenght': value} array.
        Parameters for MACD calculation, containing lenght of fast MA, lenght of slow
        MA, and lenght of signal MA, respectively. Default empty. For default values
        (12, 26, 9), input [(None, None, None)] is accepted.

    BBParams : {'lenght': value} array, lenghts of moving averages to use for Bollinger
        Bands calculation. Default empty. For default value 20, input [None] is accepted

    OBVParams : {'lenght': value}, lenght of OBV moving average. Default empty. To calculate
        OBV without moving average, input [None] is accepted

    colPrefix : string, Prefix to append to each column name. Default None

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

    if len(SMAparams) > 0:
        for param in SMAparams:
            df = calculateSMAs(df, column = param['column'], lenghts = param['lenght'])

    if len(EMAparams) > 0:
        for param in EMAparams:
            df = calculateEMAs(df, column = param['column'], lenghts = param['lenght'])

    if len(MACDParams) > 0:
        for param in MACDParams:
            df = pd.concat([df, MACD(df, n_fast = param['fast_lenght'], n_slow = param['slow_lenght'], n_signal = param['signal_lenght'])], axis = 1)

    if len(BBParams) > 0:
        for param in BBParams:
            df = pd.concat([df, BBANDS(df, n = param['lenght'])], axis = 1)

    if len(OBVParams) > 0:
        for param in OBVParams:
            df = pd.concat([df, OBV(df, n = param['lenght'])], axis = 1)

    for cols in returnCalcParams:
        if len(cols) == 1:
            df = pd.concat([df, logReturns(df[cols[0]])], axis=1)
        elif len(cols) == 2:
            df = pd.concat([df, logReturns(df[cols[0]], df[cols[1]])], axis=1)

    if deTrendParams:
        deTrend(df, column = deTrendParams['column'], window = deTrendParams['window'], model = deTrendParams['model'], weightModel = deTrendParams['weightModel'], weightModelWindow = deTrendParams['weightModelWindow'])

    if (colPrefix):
        new_names = [(column, colPrefix + '_' + column) for column in df.columns.values]
        df.rename(columns = dict(new_names), inplace=True)

    return df.dropna() if dropNan else df
