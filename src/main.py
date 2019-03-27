from src.utils import file_reader, createDataStructure
import random
from src.algorithms import knn_alg, nm_alg
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
    normalization = False

    dataSet = createDataStructure(patients, amountOfFeatures)

    ksData = kolmogorovTest(dataSet, amountOfFeatures)

    # for asdf in ksData:
    #         print("{} ({}, {})".format(asdf.getParamID(), asdf.geStatistic(), asdf.getPValue()))

    features = [9, 8, 4]
    accuracy_knn_alg = knn_alg(teachingSet, testSet, features, 5, distanceMetrics[0], normalization)
    accuracy_nm_alg = nm_alg(teachingSet, testSet, features, distanceMetrics[0], normalization)
    accuracy_nm_alg1 = nm_alg(teachingSet, testSet, features, distanceMetrics[1], normalization)

    print(accuracy_knn_alg)
    print(accuracy_nm_alg)
    print(accuracy_nm_alg1)


main()
