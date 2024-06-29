import statsmodels.stats.weightstats as stTests
import os
import sys
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
from utils.dataAnalysis import getListsFromCSV, init, handleZTests, handleWilcoxonSRTests, handleKSTests, handleJSDivergences


datasetName = "australian_credit_approval"
algorithmOne = "RF"
algorithmTwo = "GBDT"

modelType = "Classification"

algorithmOnefilename = f"Data/Classification/{datasetName}_{algorithmOne}.csv"
algorithmOneData = getListsFromCSV(algorithmOnefilename, True)

algorithmTwofilename = f"Data/Classification/{datasetName}_{algorithmTwo}.csv"
algorithmTwoData = getListsFromCSV(algorithmTwofilename, True)

print("**********")
init(algorithmOneData, algorithmTwoData)
print("**********")
handleZTests(modelType,datasetName,algorithmOne, algorithmOneData, algorithmTwo, algorithmTwoData)
print("**********")
handleWilcoxonSRTests(modelType,datasetName,algorithmOne, algorithmOneData, algorithmTwo, algorithmTwoData)
print("**********")
handleKSTests(modelType, datasetName, algorithmOne, algorithmOneData, algorithmTwo, algorithmTwoData)
print("**********")
handleJSDivergences(modelType, datasetName, algorithmOne, algorithmOneData, algorithmTwo, algorithmTwoData)
print("**********")