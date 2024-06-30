import os
import sys
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
from utils.dataAnalysis import getListsFromCSV
from utils.dataVisualization import classificationVisualization

datasetName = "taiwanese_bankruptcy_prediction"
algorithmOne = "RF"
algorithmTwo = "GBDT"

algorithmOnefilename = f"Data/Classification/{datasetName}_{algorithmOne}.csv"
algorithmOneData = getListsFromCSV(algorithmOnefilename, False)

algorithmTwofilename = f"Data/Classification/{datasetName}_{algorithmTwo}.csv"
algorithmTwoData = getListsFromCSV(algorithmTwofilename, False)

classificationUpperBound = 2

classificationVisualization(datasetName, classificationUpperBound, algorithmOne, algorithmTwo, algorithmOneData, algorithmOneData)