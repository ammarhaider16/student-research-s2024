import statsmodels.stats.weightstats as stTests
import os
import sys
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
from utils.dataAnalysis import getListsFromCSV
print("\n")

filenameGBDT = "Data/Classification/analcatdata_bondrate_GBDT.csv"
GBDTData = getListsFromCSV(filenameGBDT, False)
GBDTModelOneList = GBDTData["modelOneList"]
GBDTModelTwoList = GBDTData["modelTwoList"]
tTestGBDT = stTests.ttest_ind(GBDTModelOneList, GBDTModelTwoList)
print(f"p-value for t-Test on GBDT implementations => ", tTestGBDT[1])


filenameRF = "Data/Classification/analcatdata_bondrate_RF.csv"
RFData = getListsFromCSV(filenameRF, False)
RFModelOneList = RFData["modelOneList"]
RFModelTwoList = RFData["modelTwoList"]
tTestRF = stTests.ttest_ind(RFModelOneList, RFModelTwoList)
print(f"p-value for t-Test on RF implementations => ",tTestRF[1])





