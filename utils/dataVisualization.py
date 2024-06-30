import random

def classificationVisualization(datasetName:str,classificationUpperBound:int,algorithmOne:str,algorithmTwo:str,algorithmOneData:dict, algorithmTwoData:dict) -> None:
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Computer Modern Roman')
    plt.rc('font', size=12) 

    algorithmOneModelOneName = algorithmOneData["modelOneName"]
    algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
    algorithmOneModelOneList = algorithmOneData["modelOneList"]
    algorithmOneModelTwoList = algorithmOneData["modelTwoList"]

    algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
    algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
    algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
    algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]

    if(len(algorithmOneModelOneList)>100):
        endIndex = random.randint(101,len(algorithmOneModelOneList))
        algorithmOneModelOneList = algorithmOneModelOneList[endIndex-101:endIndex]
        algorithmOneModelTwoList = algorithmOneModelTwoList[endIndex-101:endIndex]
        algorithmTwoModelOneList = algorithmTwoModelOneList[endIndex-101:endIndex]
        algorithmTwoModelTwoList = algorithmTwoModelTwoList[endIndex-101:endIndex]

    # Plotting configuration
    fig, ax = plt.subplots(figsize=(len(algorithmOneModelOneList)/3.5,classificationUpperBound+2))

    # Plot points for each model's data from both algorithms
    ax.scatter(range(len(algorithmOneModelOneList)), algorithmOneModelOneList, label=f"{algorithmOne}/{algorithmOneModelOneName}", s=75, marker='o')
    ax.scatter(range(len(algorithmOneModelTwoList)), algorithmOneModelTwoList, label=f"{algorithmOne}/{algorithmOneModelTwoName}", s=75, marker='s')
    ax.scatter(range(len(algorithmTwoModelOneList)), algorithmTwoModelOneList, label=f"{algorithmTwo}/{algorithmTwoModelOneName}", s=75, marker='^')
    ax.scatter(range(len(algorithmTwoModelTwoList)), algorithmTwoModelTwoList, label=f"{algorithmTwo}/{algorithmTwoModelTwoName}", s=75, marker='D')

    # Set labels and title for the plot
    ax.set_yticks(list(range(classificationUpperBound)))
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"Comparison of {algorithmOne} and {algorithmTwo} on {datasetName}")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot to a file
    output_path = f"Visualization/Classification/Output/{datasetName}.png"
    plt.savefig(output_path, bbox_inches='tight')

    # Optionally, close the plot
    plt.close(fig)


def valuationVisualization(datasetName: str, algorithmOne: str, algorithmTwo: str, algorithmOneData: dict, algorithmTwoData: dict) -> None:
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', serif='Computer Modern Roman')
    plt.rc('font', size=12) 

    algorithmOneModelOneName = algorithmOneData["modelOneName"]
    algorithmOneModelTwoName = algorithmOneData["modelTwoName"]
    algorithmOneModelOneList = algorithmOneData["modelOneList"]
    algorithmOneModelTwoList = algorithmOneData["modelTwoList"]

    algorithmTwoModelOneName = algorithmTwoData["modelOneName"]
    algorithmTwoModelTwoName = algorithmTwoData["modelTwoName"]
    algorithmTwoModelOneList = algorithmTwoData["modelOneList"]
    algorithmTwoModelTwoList = algorithmTwoData["modelTwoList"]


    if(len(algorithmOneModelOneList)>50):
        endIndex = random.randint(51,len(algorithmOneModelOneList))
        algorithmOneModelOneList = algorithmOneModelOneList[endIndex-51:endIndex]
        algorithmOneModelTwoList = algorithmOneModelTwoList[endIndex-51:endIndex]
        algorithmTwoModelOneList = algorithmTwoModelOneList[endIndex-51:endIndex]
        algorithmTwoModelTwoList = algorithmTwoModelTwoList[endIndex-51:endIndex]

    # Plotting configuration
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot points for each model's data from both algorithms
    ax.scatter(range(len(algorithmOneModelOneList)), algorithmOneModelOneList, label=f"{algorithmOne}/{algorithmOneModelOneName}", s=75, marker='o')
    ax.scatter(range(len(algorithmOneModelTwoList)), algorithmOneModelTwoList, label=f"{algorithmOne}/{algorithmOneModelTwoName}", s=75, marker='s')
    ax.scatter(range(len(algorithmTwoModelOneList)), algorithmTwoModelOneList, label=f"{algorithmTwo}/{algorithmTwoModelOneName}", s=75, marker='^')
    ax.scatter(range(len(algorithmTwoModelTwoList)), algorithmTwoModelTwoList, label=f"{algorithmTwo}/{algorithmTwoModelTwoName}", s=75, marker='D')

    # Set labels and title for the plot
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"Comparison of {algorithmOne} and {algorithmTwo} on {datasetName}")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot to a file
    output_path = f"Visualization/Valuation/Output/{datasetName}.png"
    plt.savefig(output_path, bbox_inches='tight')

    # Optionally, close the plot
    plt.close(fig)