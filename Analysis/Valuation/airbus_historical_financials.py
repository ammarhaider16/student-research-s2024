import statsmodels.stats.weightstats as stTests
import os
import sys
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
from utils.dataAnalysis import getListsFromCSV
print("\n")
from utils.dataAnalysis import getAverageList

datasetName = "airbus_historical_financials"
algorithmOne = "LR"
algorithmTwo = "FNN"

algorithmOnefilename = f"Data/Valuation/{datasetName}_{algorithmOne}.csv"
algorithmOneData = getListsFromCSV(algorithmOnefilename, True)
algorithmOneModelOneName = algorithmOneData["modelOneName"]
algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
algorithmOneModelOneList = algorithmOneData["modelOneList"]
algorithmOneModelTwoList = algorithmOneData["modelTwoList"]
algorithmOneZTest = stTests.ztest(algorithmOneModelOneList, algorithmOneModelTwoList)
print(f"p-value for z-test on {algorithmOne} implementations ({algorithmOneModelOneName} and {algorithmOneModelTwoName}) => ", algorithmOneZTest[1])


algorithmTwofilename = f"Data/Valuation/{datasetName}_{algorithmTwo}.csv"
algorithmTwoData = getListsFromCSV(algorithmTwofilename, True)
algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]
algorithmTwoZTest = stTests.ztest(algorithmTwoModelOneList, algorithmTwoModelTwoList)

print(f"p-value for z-rest on {algorithmTwo} implementations ({algorithmTwoModelOneName} and {algorithmTwoModelTwoName}) => ", algorithmTwoZTest[1])

with open("Analysis/Valuation/ValuationImplementationZTests.csv","a") as file:
    file.write(f"\n{datasetName},{algorithmOneZTest[1]}, {algorithmTwoZTest[1]}")

algorithmOneAverageList = getAverageList(algorithmOneModelOneList, algorithmOneModelTwoList)
algorithmTwoAverageList = getAverageList(algorithmTwoModelOneList, algorithmTwoModelTwoList)
acrossAlgorithmZTest = stTests.ztest(algorithmOneAverageList, algorithmTwoAverageList)

print(f"p-value for z-rest on {algorithmOne} and {algorithmTwo} => ", acrossAlgorithmZTest[1])

with open("Analysis/Valuation/ValuationAlgorithmZTests.csv","a") as file:
    file.write(f"\n{datasetName},{acrossAlgorithmZTest[1]}")
