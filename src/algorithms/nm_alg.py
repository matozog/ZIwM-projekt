import collections
import scipy
from src.utils.data_structure import prepareDataSet
import numpy as np


def calculateSizeOfEachClass(trainDataLabelFeatures):
    counter = collections.Counter(trainDataLabelFeatures)
    sizeOfB, sizeOfM = counter['B'], counter['M']
    return sizeOfM, sizeOfB


def createMeanVectors(nmForMClass, nmForBClass, trainDataLabelFeatures):
    sizeOfM, sizeOfB  = calculateSizeOfEachClass(trainDataLabelFeatures)
    for feature in range(0, nmForMClass.__len__()):
        nmForMClass[feature] /= sizeOfM
        nmForBClass[feature] /= sizeOfB
    return nmForMClass, nmForBClass


def calculateNearestMean(trainDataFeatures, trainDataLabelFeatures, amountOfFeatures):
    nmForMClass = [0.0]*amountOfFeatures
    nmForBClass = [0.0]*amountOfFeatures

    for feature in range(0, amountOfFeatures):
        for patient in range(0, trainDataFeatures.__len__()):
            if trainDataLabelFeatures[patient] == 'M':
                nmForMClass[feature] += float(trainDataFeatures[patient][feature])
            else:
                nmForBClass[feature] += float(trainDataFeatures[patient][feature])

    return createMeanVectors(nmForMClass, nmForBClass, trainDataLabelFeatures)


def assignToClass(testDataFeatures, nearestMeanForBClass, nearestMeanForMClass, distanceMetrics):
    results = []

    for patient in testDataFeatures:
        float_patient_list = [float(i) for i in patient]
        if distanceMetrics == 'manhattan':
            distanceToMClass = scipy.spatial.distance.cityblock(float_patient_list, nearestMeanForMClass)
            distanceToBClass = scipy.spatial.distance.cityblock(float_patient_list, nearestMeanForBClass)
        else:
            distanceToMClass = scipy.spatial.distance.euclidean(float_patient_list, nearestMeanForMClass)
            distanceToBClass = scipy.spatial.distance.euclidean(float_patient_list, nearestMeanForBClass)

        if distanceToMClass < distanceToBClass:
            results.append('M')
        else:
            results.append('B')

    return results


# def normalizingData(trainDataFeatures):
#     pprint.pprint(trainDataFeatures)
#     for patient in range(0,trainDataFeatures.__len__()):
#         tra = numpy.linalg.norm(patientFeatures)
#     pprint.pprint(trainDataFeatures)


def nm_alg(teachingSet, testSet, features, distanceMetrics, normalization):

    trainDataFeatures, trainDataLabelFeatures = prepareDataSet(teachingSet, features, normalization)
    testDataFeatures, testDataLabelFeatures = prepareDataSet(testSet, features, normalization)

    meanForMClass, meanForBClass = calculateNearestMean(trainDataFeatures, trainDataLabelFeatures, features.__len__())

    classifiedData = assignToClass(testDataFeatures, meanForBClass, meanForMClass, distanceMetrics)

    classifiedNpArray = np.array(classifiedData)
    testDataLabelNpArray = np.array(testDataLabelFeatures)

    accuracy = np.mean(classifiedNpArray == testDataLabelNpArray)

    return accuracy
