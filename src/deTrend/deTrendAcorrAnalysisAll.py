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
# savePath = '../ns-results/src/deTrend'
stocks = os.listdir(dataPath + '/stocks')

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

func = partial(deTrendAcorrGreedySearch, dataPath = dataPath, savePath = savePath, bot = bot)
# Start Parallel processing
num_processes = multiprocessing.cpu_count()
p = multiprocessing.Pool(processes=num_processes)
results = p.map(func, stocks)
p.close()
p.join()

#concatenate and save all results
allResults = {}
for r in results:
    allResults[r['asset']] = {'optimalTrendSamples': r['optimalTrendSamples'], 'optimalAcorrSamples': r['optimalAcorrSamples'], 'optimalRMSE': r['optimalRMSE']}
joblib.dump(allResults, '{}/allAcorrSweepResults.pkl'.format(savePath + '/Variables'))
