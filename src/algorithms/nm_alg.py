from sklearn import neighbors
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from src.utils.data_structure import prepareDataSet
import numpy as np


def nm_alg(teachingSet, testSet, features, distanceMetrics, normalization):

    trainDataFeatures, trainDataLabelFeatures = prepareDataSet(teachingSet, features, normalization)
    testDataFeatures, testDataLabelFeatures = prepareDataSet(testSet, features, normalization)

    classifier = neighbors.NearestCentroid(metric=distanceMetrics, shrink_threshold=None)

    classifier.fit(trainDataFeatures, trainDataLabelFeatures)
    predictions = classifier.predict(testDataFeatures)

    accuracy = accuracy_score(testDataLabelFeatures, predictions)

    # accuracy_balanced = balanced_accuracy_score(testDataLabelFeatures, predictions)
    # accuracy_confusion_matrix = confusion_matrix(testDataLabelFeatures, predictions)
    # tn, fp, fn, tp = confusion_matrix(testDataLabelFeatures, predictions).ravel()
    #
    # print(accuracy)
    # print(accuracy_balanced)
    # print(accuracy_confusion_matrix)
    # print(tn, fp, fn, tp)

    return accuracy
