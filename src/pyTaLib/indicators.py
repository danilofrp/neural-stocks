import numpy
from pandas import *
from math import *

#Moving Average
def SMA(df, column, n):
    """ Simple Moving Average
    Parameters
    ----------
    df : pandas.DataFrame object
    column : string, DataFrame column desired for calculation of moving average
    n : int, lenght of moving average

    Returns
    ----------
    SMA_[n] : pandas.Series, simple moving average (values before [n] samples will be NaN)
    """
    return Series(df[column].rolling(window = n, center = False).mean(), name = column + '_SMA' + str(n))

#Exponential Moving Average
def EMA(df, column, n):
    """ Exponential Moving Average
    Parameters
    ----------
    df : pandas.DataFrame object
    column : string, DataFrame column desired for calculation of moving average
    n : int, lenght of moving average

    Returns
    ----------
    EMA_[n] : pandas.Series, exponential moving average (values before [n] samples will be NaN)
    """
    return Series(Series.ewm(df[column], span = n, min_periods = n - 1).mean(), name = column + '_EMA' + str(n))

#Momentum
def MOM(df, column, n):
    return Series(df[column].diff(n), name = column + '_Momentum_' + str(n))

#Rate of Change
def ROC(df, column,  n):
    M = df[column].diff(n - 1)
    N = df[column].shift(n - 1)
    ROC = Series(M / N, name = column + '_ROC_' + str(n))
    return ROC

#Average True Range
def ATR(df, n):
    index = df.index.values
    df.reset_index()
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = Series(TR_l)
    ATR = Series(Series.ewm(TR_s, span = n, min_periods = n).mean(), index = index, name = 'ATR_' + str(n))
    return ATR

#Bollinger Bands
def BBANDS(df, n = 20):
    """ Bollinger Bands
    Parameters
    ----------
    df : pandas.DataFrame object ('Close' colum assumed to be present)
    n : int, lenght of moving average, default 20

    Returns
    ----------
    BBANDS : pandas.DataFrame, containing the columns (where [n] is the given input):
                - BollingerMA_[n]: simple moving average of [n] periods
                - BollingerUpper_[n]: upper band limit (MA + 2 * STD)
                - BollingerLower_[n]: lower band limit (MA - 2 * STD)
                - Bollinger%b_[n]: relative position of Close to band limits (-1 = lower, 0 = MA, 1 = upper)
    """
    MA = df['Close'].rolling(window = n, center = False).mean()
    MSD = df['Close'].rolling(window = n, center = False).std()
    BMA = Series(MA, name = 'BollingerMA_' + str(n))
    BUp = Series(BMA + 2 * MSD, name = 'BollingerUpper_' + str(n))
    BLow = Series(BMA - 2 * MSD, name = 'BollingerLower_' + str(n))
    bp = (df['Close'] - MA) / (2 * MSD)
    Bp = Series(bp, name = 'Bollinger%b_' + str(n))
    BB = concat([BMA, BUp, BLow, Bp], axis = 1)
    return BB

#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = Series(2 * PP - df['Low'])
    S1 = Series(2 * PP - df['High'])
    R2 = Series(PP + df['High'] - df['Low'])
    S2 = Series(PP - df['High'] + df['Low'])
    R3 = Series(df['High'] + 2 * (PP - df['Low']))
    S3 = Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = DataFrame(psr)
    return PSR

#Stochastic oscillator %K
def STOK(df):
    return Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')

#Stochastic oscillator %D
def STOD(df, n):
    SOk = Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    SOd = Series(Series.ewm(SOk, span = n, min_periods = n - 1).mean(), name = 'SO%d_' + str(n))
    return SOd

#Trix
def TRIX(df, n):
    index = df.index.values
    df.reset_index()
    EX1 = Series.ewm(df['Close'], span = n, min_periods = n - 1).mean()
    EX2 = Series.ewm(EX1, span = n, min_periods = n - 1).mean()
    EX3 = Series.ewm(EX2, span = n, min_periods = n - 1).mean()
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = Series(ROC_l, index = index, name = 'Trix_' + str(n))
    return Trix

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    index = df.index.values
    df.reset_index()
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = Series(TR_l)
    ATR = Series(Series.ewm(TR_s, span = n, min_periods = n).mean())
    UpI = Series(UpI)
    DoI = Series(DoI)
    PosDI = Series(Series.ewm(UpI, span = n, min_periods = n - 1).mean() / ATR)
    NegDI = Series(Series.ewm(DoI, span = n, min_periods = n - 1).mean() / ATR)
    ADX = Series(Series.ewm(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1).mean(), index = index, name = 'ADX_' + str(n) + '_' + str(n_ADX))
    return ADX

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast = 12, n_slow = 26, n_signal = 9):
    """ Moving Average Convergence/Divergence
    Parameters
    ----------
    df : pandas.DataFrame object ('Close' colum assumed to be present)
    n_fast : int, lenght of fast moving average, default 12
    n_slow : int, lenght of slow moving average, default 26
    n_signal : int, lenght of signal moving average, default 9

    Returns
    ----------
    MACD : pandas.DataFrame, containing the columns (where [n_...] are the given inputs):
                - MACD_[n_fast]_[n_slow]: Difference between [n_fast] EMA and [n_slow] EMA
                - MACDsignal_[n_fast]_[n_slow]: [n_signal]-period moving average of MACD
                - MACDdiff_[n_fast]_[n_slow]: difference between MACD and MACDsignal
    """
    EMAfast = Series(Series.ewm(df['Close'], span = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = Series(Series.ewm(df['Close'], span = n_slow, min_periods = n_slow - 1).mean())
    MACD = Series(EMAfast - EMAslow, name = 'MACD_{}_{}_{}'.format(n_fast, n_slow, n_signal))
    MACDsign = Series(Series.ewm(MACD, span = n_signal, min_periods = n_signal - 1).mean(), name = 'MACDsignal_{}_{}_{}'.format(n_fast, n_slow, n_signal))
    MACDdiff = Series(MACD - MACDsign, name = 'MACDdiff_{}_{}_{}'.format(n_fast, n_slow, n_signal))
    macd = pandas.concat([MACD, MACDsign, MACDdiff], axis = 1)
    return macd

#Mass Index
def MassI(df):
    Range = df['High'] - df['Low']
    EX1 = Series.ewm(Range, span = 9, min_periods = 8).mean()
    EX2 = Series.ewm(EX1, span = 9, min_periods = 8).mean()
    Mass = EX1 / EX2
    MassI = Series(rolling_sum(Mass, 25), name = 'Mass Index')
    return MassI

#Vortex Indicator
def Vortex(df, n):
    index = df.index.values
    df.reset_index()
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = Series(rolling_sum(Series(VM), n) / rolling_sum(Series(TR), n), index = index, name = 'Vortex_' + str(n))
    return VI

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = Series(rolling_sum(ROC1, n1) + rolling_sum(ROC2, n2) * 2 + rolling_sum(ROC3, n3) * 3 + rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    return KST

#Relative Strength Index
def RSI(df, n):
    index = df.index.values
    df.reset_index()
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = Series(UpI)
    DoI = Series(DoI)
    PosDI = Series(Series.ewm(UpI, span = n, min_periods = n - 1).mean())
    NegDI = Series(Series.ewm(DoI, span = n, min_periods = n - 1).mean())
    RSI = Series(PosDI / (PosDI + NegDI), index = index, name = 'RSI_' + str(n))
    return RSI

#True Strength Index
def TSI(df, r, s):
    M = Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = Series(Series.ewm(M, span = r, min_periods = r - 1).mean())
    aEMA1 = Series(Series.ewm(aM, span = r, min_periods = r - 1).mean())
    EMA2 = Series(Series.ewm(EMA1, span = s, min_periods = s - 1).mean())
    aEMA2 = Series(Series.ewm(aEMA1, span = s, min_periods = s - 1).mean())
    TSI = Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))
    return TSI

#Accumulation/Distribution
def ACCDIST(df, n):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    return AD

#Chaikin Oscillator
def Chaikin(df):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = Series(Series.ewm(ad, span = 3, min_periods = 2).mean() - Series.ewm(ad, span = 10, min_periods = 9).mean(), name = 'Chaikin')
    return Chaikin

#Money Flow Index and Ratio
def MFI(df, n):
    index = df.index.values
    df.reset_index()
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = Series(PosMF / TotMF)
    MFI = Series(rolling_mean(MFR, n), index = index, name = 'MFI_' + str(n))
    return MFI

#On-balance Volume
def OBV(df, n = None):
    """ On Balance Volume
    Parameters
    ----------
    df : pandas.DataFrame object ('Close' and 'Volume' colum assumed to be present)
    n : int, lenght of moving average, default None

    Returns
    ----------
    OBV : pandas.DataFrame, containing the columns (where [n] is the given input):
                - OBV: On Balance Volume
                - OBV[n]: simple n-sample moving average of OBV, if n is given
    """
    OBV = [np.nan]
    i = 1
    while i < len(df):
        signal = sign(df.get_value(df.index[i], 'Close') - df.get_value(df.index[i - 1], 'Close'))
        OBV.append(signal * df.get_value(df.index[i], 'Volume'))
        i += 1
    OBV = Series(OBV, index = df.index, name = 'OBV').cumsum()
    if n:
        OBV_ma = Series(OBV.rolling(window = n, center = False).mean(), name = 'OBV{}'.format(n))
        return concat([OBV, OBV_ma], axis = 1)
    else:
        return OBV

#Force Index
def FORCE(df, n):
    return Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))

#Ease of Movement
def EOM(df, n):
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    EoM_ma = Series(rolling_mean(EoM, n), name = 'EoM_' + str(n))
    return EoM_ma

#Commodity Channel Index
def CCI(df, n):
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = Series((PP - rolling_mean(PP, n)) / rolling_std(PP, n), name = 'CCI_' + str(n))
    return CCI

#Coppock Curve
def COPP(df, n):
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = Series(Series.ewm(ROC1 + ROC2, span = n, min_periods = n).mean(), name = 'Copp_' + str(n))
    return Copp

#Keltner Channel
def KELCH(df, n):
    KelChM = Series(rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), name = 'KelChM_' + str(n))
    KelChU = Series(rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), name = 'KelChU_' + str(n))
    KelChD = Series(rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), name = 'KelChD_' + str(n))
    KelCh = concat([KelChM, KelChU, KelChD], axis = 1)
    return KelCh

#Ultimate Oscillator
def ULTOSC(df):
    index = df.index.values
    df.reset_index()
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        TR_l.append(TR)
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = Series((4 * rolling_sum(Series(BP_l), 7) / rolling_sum(Series(TR_l), 7)) + (2 * rolling_sum(Series(BP_l), 14) / rolling_sum(Series(TR_l), 14)) + (rolling_sum(Series(BP_l), 28) / rolling_sum(Series(TR_l), 28)), index = index, name = 'Ultimate_Osc')
    return UltO

#Donchian Channel
def DONCH(df, n):
    index = df.index.values
    df.reset_index()
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = Series(DC_l, index = index, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    return DonCh

#Standard Deviation
def STDDEV(df, column, n):
    return Series(df[column].rolling(window = n, center = False).std(), name = column + '_STD' + str(n))

sign = lambda a: int(a>0) - int(a<0)
