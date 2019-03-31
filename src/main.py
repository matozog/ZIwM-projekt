import csv
from functools import reduce
from src.models import Alghorithm
from src.utils import file_reader, createDataStructure
import random
from src.utils import kolmogorovTest


def createTeachingAndTestSets(patients):
    teachingSet = []
    testSet = []
    halfOfPatients = int(patients.__len__()/2)
    drawnNumbers = []

    while teachingSet.__len__() < halfOfPatients:
        randNumber = random.randint(0, patients.__len__()-1)
        if randNumber not in drawnNumbers:
            teachingSet.append(patients[randNumber])
            drawnNumbers.append(randNumber)

    for i in range(patients.__len__()):
        if i not in drawnNumbers:
            testSet.append(patients[i])

    return teachingSet, testSet

def main():
    file_name = "../resources/wdbc.data"
    size_of_data = 32
    amountOfFeatures = 10
    patients = file_reader.loadDataFromFile(file_name, size_of_data)
    teachingSet, testSet = createTeachingAndTestSets(patients)
    distanceMetrics = ['euclidean', 'manhattan']
    normalization = [True, False]
    algorithms = [Alghorithm('NM', 'nm_alg'), Alghorithm('NN_1', 'knn_alg',  1), Alghorithm('NN_5','knn_alg', 5), Alghorithm('NN_9','knn_alg', 9)]

    dataSet = createDataStructure(patients, amountOfFeatures)

    ksData = kolmogorovTest(dataSet, amountOfFeatures)


    # for asdf in ksData:
    #         print("{} ({}, {})".format(asdf.getParamID(), asdf.geStatistic(), asdf.getPValue()))

    # features = [7]
    # accuracy_knn_alg = knn_alg(teachingSet, testSet, features, 5, distanceMetrics[0], False)
    # accuracy_knn_alg1 = knn_alg(teachingSet, testSet, features, 5, distanceMetrics[0], True)
    # accuracy_nm_alg = nm_alg(teachingSet, testSet, features, distanceMetrics[0], False)
    # accuracy_nm_alg1 = nm_alg(teachingSet, testSet, features, distanceMetrics[0], True)

    # print(accuracy_knn_alg)
    # print(accuracy_knn_alg1)
    # print(accuracy_nm_alg)
    # print(accuracy_nm_alg1)


main()
