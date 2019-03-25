from sklearn import neighbors
from sklearn.metrics import accuracy_score


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


def knn_alg(teachingSet, testSet, features):
    trainDataFeatures, trainDataLabelFeatures = prepareDataSet(teachingSet, features)

    testDataFeatures, testDataLabelFeatures = prepareDataSet(testSet, features)

    # testDataLabelFeatures.astype(int)
    # trainDataLabelFeatures.astype(int)

    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(trainDataFeatures, trainDataLabelFeatures)
    predictions = classifier.predict(testDataFeatures)
    score = accuracy_score(testDataLabelFeatures, predictions)
    # print(score)

    return score
