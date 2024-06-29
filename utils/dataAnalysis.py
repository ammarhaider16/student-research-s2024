import statsmodels.stats.weightstats as sms
import scipy.stats as sps
import scipy.spatial.distance as spd
import numpy as np

def getListsFromCSV(filename: str, isFloat:bool = True) -> dict:
    modelOneName = ""
    modelTwoName = ""
    modelOneList = []
    modelTwoList = []

    with open(filename, "r") as file:
        title = file.readline().strip().split(",")
        modelOneName = title[0]
        modelTwoName = title[1]
        
        for line in file.readlines():
            lineList = line.strip().split(",")
            if isFloat:
                modelOneList.append(float(lineList[0]))
                modelTwoList.append(float(lineList[1]))
            else:
                modelOneList.append(int(lineList[0]))
                modelTwoList.append(int(lineList[1]))

    return {
        "modelOneName": modelOneName,
        "modelTwoName": modelTwoName,
        "modelOneList":modelOneList,
        "modelTwoList":modelTwoList
    }

def getAverageList(listOne:list, listTwo:list) -> list:
    if len(listOne) != len(listTwo):
        raise Exception("Lists are not equal in length!")

    averageList = []
    for i in range(len(listOne)):
        average = (listOne[i]+listTwo[i])/2
        averageList.append(average)

    return averageList

def init(algorithmOneData:dict, algorithmTwoData:dict) -> None:
    algorithmOneModelOneList = algorithmOneData["modelOneList"]
    algorithmOneModelTwoList = algorithmOneData["modelTwoList"]

    algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
    algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]

    print(f"len algorithmOneList => {len(algorithmOneModelOneList)}")
    print(f"len algorithmTwoList => {len(algorithmTwoModelOneList)}")

def handleZTests(modelType:str,datasetName:str, algorithmOne:str,algorithmOneData:dict, algorithmTwo:str,algorithmTwoData:dict) -> None:
    algorithmOneModelOneName = algorithmOneData["modelOneName"]
    algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
    algorithmOneModelOneList = algorithmOneData["modelOneList"]
    algorithmOneModelTwoList = algorithmOneData["modelTwoList"]

    algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
    algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
    algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
    algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]

    algorithmOneAverageList = getAverageList(algorithmOneModelOneList, algorithmOneModelTwoList)
    algorithmTwoAverageList = getAverageList(algorithmTwoModelOneList, algorithmTwoModelTwoList)

    
    algorithmOneZTest = sms.ztest(algorithmOneModelOneList, algorithmOneModelTwoList)
    print(f"p-value for z-test on {algorithmOne} implementations ({algorithmOneModelOneName} and {algorithmOneModelTwoName}) => ", algorithmOneZTest[1])
    algorithmTwoZTest = sms.ztest(algorithmTwoModelOneList, algorithmTwoModelTwoList)
    acrossAlgorithmZTest = sms.ztest(algorithmOneAverageList, algorithmTwoAverageList)

    
    print(f"p-value for z-test on {algorithmTwo} implementations ({algorithmTwoModelOneName} and {algorithmTwoModelTwoName}) => ", algorithmTwoZTest[1])
    print(f"p-value for z-test on {algorithmOne} and {algorithmTwo} => ", acrossAlgorithmZTest[1])


    with open(f"Analysis/{modelType}/Output/{modelType}ImplementationZTests.csv","a") as file:
        file.write(f"\n{datasetName},{algorithmOneZTest[1]},{algorithmTwoZTest[1]}")


    with open(f"Analysis/{modelType}/Output/{modelType}AlgorithmZTests.csv","a") as file:
        file.write(f"\n{datasetName},{acrossAlgorithmZTest[1]}")

def handleWilcoxonSRTests(modelType:str,datasetName:str, algorithmOne:str,algorithmOneData:dict, algorithmTwo:str,algorithmTwoData:dict) -> None:
    algorithmOneModelOneName = algorithmOneData["modelOneName"]
    algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
    algorithmOneModelOneList = algorithmOneData["modelOneList"]
    algorithmOneModelTwoList = algorithmOneData["modelTwoList"]

    algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
    algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
    algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
    algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]

    algorithmOneWRSTest = sps.wilcoxon(algorithmOneModelOneList, algorithmOneModelTwoList)
    print(f"p-value for wilcoxon sr test on {algorithmOne} implementations ({algorithmOneModelOneName} and {algorithmOneModelTwoName}) => ", algorithmOneWRSTest.pvalue)

    algorithmTwoWRSTest = sps.wilcoxon(algorithmTwoModelOneList, algorithmTwoModelTwoList)
    print(f"p-value for wilcoxon sr test on {algorithmTwo} implementations ({algorithmTwoModelOneName} and {algorithmTwoModelTwoName}) => ", algorithmTwoWRSTest.pvalue)


    with open(f"Analysis/{modelType}/Output/{modelType}ImplementationWilcoxonSRTests.csv","a") as file:
        file.write(f"\n{datasetName},{algorithmOneWRSTest.pvalue},{algorithmTwoWRSTest.pvalue}")

    algorithmOneAverageList = getAverageList(algorithmOneModelOneList, algorithmOneModelTwoList)
    algorithmTwoAverageList = getAverageList(algorithmTwoModelOneList, algorithmTwoModelTwoList)
    acrossAlgorithmWRSTest = sps.wilcoxon(algorithmOneAverageList, algorithmTwoAverageList)

    print(f"p-value for wilcoxon sr test on {algorithmOne} and {algorithmTwo} => ", acrossAlgorithmWRSTest.pvalue)

    with open(f"Analysis/{modelType}/Output/{modelType}AlgorithmWilcoxonSRTests.csv","a") as file:
        file.write(f"\n{datasetName},{acrossAlgorithmWRSTest.pvalue}")

def handleKSTests(modelType:str,datasetName:str, algorithmOne:str,algorithmOneData:dict, algorithmTwo:str,algorithmTwoData:dict) -> None:
    algorithmOneModelOneName = algorithmOneData["modelOneName"]
    algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
    algorithmOneModelOneList = algorithmOneData["modelOneList"]
    algorithmOneModelTwoList = algorithmOneData["modelTwoList"]

    algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
    algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
    algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
    algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]

    algorithmOneKSTest = sps.ks_2samp(algorithmOneModelOneList, algorithmOneModelTwoList)
    print(f"p-value for ks test on {algorithmOne} implementations ({algorithmOneModelOneName} and {algorithmOneModelTwoName}) => ", algorithmOneKSTest.pvalue)

    algorithmTwoKSTest = sps.ks_2samp(algorithmTwoModelOneList, algorithmTwoModelTwoList)
    print(f"p-value for ks test on {algorithmTwo} implementations ({algorithmTwoModelOneName} and {algorithmTwoModelTwoName}) => ", algorithmTwoKSTest.pvalue)


    with open(f"Analysis/{modelType}/Output/{modelType}ImplementationKSTests.csv","a") as file:
        file.write(f"\n{datasetName},{algorithmOneKSTest.pvalue},{algorithmTwoKSTest.pvalue}")

    algorithmOneAverageList = getAverageList(algorithmOneModelOneList, algorithmOneModelTwoList)
    algorithmTwoAverageList = getAverageList(algorithmTwoModelOneList, algorithmTwoModelTwoList)
    acrossAlgorithmKSTest = sps.ks_2samp(algorithmOneAverageList, algorithmTwoAverageList)

    print(f"p-value for ks test on {algorithmOne} and {algorithmTwo} => ", acrossAlgorithmKSTest.pvalue)

    with open(f"Analysis/{modelType}/Output/{modelType}AlgorithmKSTests.csv","a") as file:
        file.write(f"\n{datasetName},{acrossAlgorithmKSTest.pvalue}")

def normalize_distribution(distribution: list) -> np.ndarray:
    epsilon = 1e-10
    distribution = np.array(distribution, dtype=np.float64)
    if np.sum(distribution) == 0:
        raise ValueError("The sum of the distribution is zero, cannot normalize.")
    distribution = distribution / np.sum(distribution)  # Normalize to sum to 1
    distribution = distribution + epsilon  # Add epsilon to avoid zeros
    return distribution

def handleJSDivergences(modelType:str,datasetName:str, algorithmOne:str,algorithmOneData:dict, algorithmTwo:str,algorithmTwoData:dict) -> None:
    base = 2

    algorithmOneModelOneName = algorithmOneData["modelOneName"]
    algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
    algorithmOneModelOneList = normalize_distribution(algorithmOneData["modelOneList"])
    algorithmOneModelTwoList = normalize_distribution(algorithmOneData["modelTwoList"])

    algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
    algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
    algorithmTwoModelOneList = normalize_distribution(algorithmTwoData["modelOneList"])
    algorithmTwoModelTwoList = normalize_distribution(algorithmTwoData["modelTwoList"])

    algorithmOneJSDivergence = spd.jensenshannon(algorithmOneModelOneList, algorithmOneModelTwoList,base)
    print(f"jensen shannon divergence for {algorithmOne} implementations ({algorithmOneModelOneName} and {algorithmOneModelTwoName}) => ", algorithmOneJSDivergence)

    algorithmTwoJSDivergence = spd.jensenshannon(algorithmTwoModelOneList, algorithmTwoModelTwoList,base)
    print(f"jensen shannon divergence for {algorithmTwo} implementations ({algorithmTwoModelOneName} and {algorithmTwoModelTwoName}) => ", algorithmTwoJSDivergence)


    with open(f"Analysis/{modelType}/Output/{modelType}ImplementationJSDivergences.csv","a") as file:
        file.write(f"\n{datasetName},{algorithmOneJSDivergence},{algorithmTwoJSDivergence}")

    algorithmOneAverageList = getAverageList(algorithmOneModelOneList, algorithmOneModelTwoList)
    algorithmTwoAverageList = getAverageList(algorithmTwoModelOneList, algorithmTwoModelTwoList)
    acrossAlgorithmJSDivergence = spd.jensenshannon(algorithmOneAverageList, algorithmTwoAverageList,base)

    print(f"jensen shannon divergence for {algorithmOne} and {algorithmTwo} => ", acrossAlgorithmJSDivergence)

    with open(f"Analysis/{modelType}/Output/{modelType}AlgorithmJSDivergences.csv","a") as file:
        file.write(f"\n{datasetName},{acrossAlgorithmJSDivergence}")
