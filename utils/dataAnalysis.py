

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
        raise Exception("Listd are not equal in length!")

    averageList = []
    for i in range(len(listOne)):
        average = (listOne[i]+listTwo[i])/2
        averageList.append(average)

    return averageList
