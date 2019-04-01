import scipy

#   finalDataSet = [[parameters patient1], [ param patient2], ....]
#   finalDataLabelSet = [[result patient1], [result patient2], ...]


def prepareDataSet(dataSet, features, normalization):
    finalDataSet = []
    finalLabelSet = []
    for patient in dataSet:
        featureSet = []
        finalLabelSet.append(patient.getCancerType())
        for x in features:
            featureSet.append(patient.getInputValues()[x])
        float_featureSet = [float(i) for i in featureSet]
        if normalization:
            normalFeatureSet = normalizeVector(float_featureSet)
            finalDataSet.append(normalFeatureSet)
        else:
            finalDataSet.append(float_featureSet)

    return finalDataSet, finalLabelSet


def normalizeVector(vector):
    normalFeatureSet = []
    vectorLength = scipy.linalg.norm(vector)
    for number in vector:
        if vectorLength != 0.0:
            normalFeatureSet.append(number / vectorLength)
        else:
            normalFeatureSet.append(0)
    return normalFeatureSet


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