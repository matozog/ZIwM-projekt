from src.ksType import ksType
from src.utils import file_reader
from scipy.stats import ks_2samp
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import random
import pprint
import numpy as np

def createTeachingAndTestSets(patients):
    teachingSet = []
    testSet = []
    halfOfPatients = int(patients.__len__()/2)
    drawnNumbers = []

    while teachingSet.__len__() < halfOfPatients:
        randNumber = random.randint(0, patients.__len__()-1)
        if randNumber == 569:
            print("lol")
        if randNumber not in drawnNumbers:
            teachingSet.append(patients[randNumber])
            drawnNumbers.append(randNumber)

    for i in range(patients.__len__()):
        if i not in drawnNumbers:
            testSet.append(patients[i])

    return teachingSet, testSet

def prepareDataSet(dataSet, features=[]):
    finalSet = []
    finalLabelSet = []

    for data in dataSet:
        featureSet = []
        finalLabelSet.append(data.getCancerType())
        for x in features:
            featureSet.append(data.getInputValues()[x])
        finalSet.append(featureSet)
    return finalSet, finalLabelSet

def main():
    file_name = "../resources/wdbc.data"
    size_of_data = 32
    patients = file_reader.loadDataFromFile(file_name, size_of_data)
    teachingSet, testSet = createTeachingAndTestSets(patients)
    dataSet = {}
    dataSet["M"] = {}
    dataSet["B"] = {}
    ksData = []

    for x in range(0, 10):
        dataSet["B"][x] = []
        dataSet["M"][x] = []

    # print(patients.__len__())
    for patient in patients:
        for x in range(0, 10):
            val = patient.getInputValues()[x]
            dataSet[patient.getCancerType()][x].append(val)
    for a in range(0, 10):
        x, d = ks_2samp(dataSet["M"][a], dataSet["B"][a])
        ksData.append(ksType(a, x, d))

    ksData.sort(key=lambda val : val.geStatistic())

    
    # for asdf in ksData:
    #     #     print("{} ({}, {})".format(asdf.getParamID(), asdf.geStatistic(), asdf.getPValue()))

    # nmAlg(ksData[0], teachingSet, testSet, dataSet)

    # iris = datasets.load_iris()
    # x = iris.data
    # y = iris.target
    #
    # print(x)
    # print(y)

    trainDataFeatures, trainDataLabelFeatures = prepareDataSet(teachingSet, [9, 8, 4, 0])
    # pprint.pprint(trainDataFeatures)
    # pprint.pprint(trainDataLabelFeatures)

    testDataFeatures, testDataLabelFeatures = prepareDataSet(testSet, [9, 8, 4, 0])
    # pprint.pprint(testDataFeatures)
    # pprint.pprint(testDataLabelFeatures)

#    testDataLabelFeatures.astype(int)
 #   trainDataLabelFeatures.astype(int)

    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(trainDataFeatures, trainDataLabelFeatures)
    predictions = classifier.predict(testDataFeatures)
    # print(accuracy_score(testDataLabelFeatures, predictions))

    # classifier.fit([ [teachingSet[9].getInputValues()[ksData[0].getParamID()]], [teachingSet[10].getInputValues()[ksData[0].getParamID()]]],
    #                [[0], [1]])
    # predictions = classifier.predict(testSet[0].getInputValues()[9])
    #
    # print(accuracy_score(testSet[0].getCancerType(), predictions))
# def nmAlg(param, teachingSet, trainSet, dataSet):


main()

