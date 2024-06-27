def modelOutputToCSV(modelOneName:str, modelOneOutputList:list, modelTwoName:str, modelTwoOutputList:list, filePath:str):

    if len(modelOneOutputList) != len(modelTwoOutputList):
        print("Output lists do not have the same length!")
    
    with open(filePath, 'w') as file: 
        file.write(f"{modelOneName},{modelTwoName}\n") 
        for index in range(len(modelOneOutputList)):
            file.write(f"{modelOneOutputList[index]},{modelTwoOutputList[index]}\n")

    return "Created .csv file with model output!"