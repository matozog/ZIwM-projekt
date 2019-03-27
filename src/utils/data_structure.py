
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