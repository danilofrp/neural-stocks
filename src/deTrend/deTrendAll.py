# <editor-fold> IMPORTS
from __future__ import print_function
import sys, os
sys.path.append('..')
import multiprocessing
from functools import partial
from sklearn.externals import joblib
from neuralstocks.utils import *
from neuralstocks.deTrendSweeps import *
from neuralstocks.preprocessing import *
from messaging.telegrambot import Bot
# </editor-fold>

bot = Bot('neuralStocks')
dataPath, savePath = setPaths(__file__)
# dataPath = '../data'
# savePath = '../ns-results/data/preprocessed/diario'
stocks = os.listdir(dataPath + '/stocks')

func = partial(deTrendOptimal, dataPath = dataPath, savePath = savePath, bot = bot)
# Start Parallel processing
num_processes = multiprocessing.cpu_count()
p = multiprocessing.Pool(processes=num_processes)
results = p.map(func, stocks)
p.close()
p.join()
