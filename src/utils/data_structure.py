
def prepareDataSet(dataSet, features):
    finalDataSet = []
    finalLabelSet = []

    for patient in dataSet:
        featureSet = []
        finalLabelSet.append(patient.getCancerType())
        for x in features:
            featureSet.append(patient.getInputValues()[x])
        finalDataSet.append(featureSet)
    return finalDataSet, finalLabelSet


def createDataStructure(patients, amountOfFeatures):
    dataSet = {}
    dataSet["M"] = {}
    dataSet["B"] = {}

    for x in range(0, amountOfFeatures):
        dataSet["B"][x] = []
        dataSet["M"][x] = []

    for patient in patients:
        for x in range(0, amountOfFeatures):
            val = patient.getInputValues()[x]
            dataSet[patient.getCancerType()][x].append(val)
    return dataSet