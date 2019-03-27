import scipy


def prepareDataSet(dataSet, features, normalization):
    finalDataSet = []
    finalLabelSet = []
    for patient in dataSet:
        featureSet = []
        finalLabelSet.append(patient.getCancerType())
        for x in features:
            featureSet.append(patient.getInputValues()[x])
        if(normalization):
            float_featureSet = [float(i) for i in featureSet]
            normalFeatureSet = [number/scipy.linalg.norm(float_featureSet) for number in float_featureSet]
            finalDataSet.append(normalFeatureSet)
        else:
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