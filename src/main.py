import csv
from functools import reduce

from numpy import std, mean, asarray

from src.algorithms import nm_alg, knn_alg
from src.models import Alghorithm
from src.models.alghoritm_parameters import AlgParameters
from src.models.classification_result import ClassificationResult
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


def createReasearches(patients, algorithms,ksData ,algParameters):

    halo = {}
    for metric in algParameters.getMetrics():
        for disMetric in algParameters.getDistanceMetrics():
            for norm in algParameters.getNormalization():
                halo[metric+disMetric+str(norm)] = ClassificationResult(ksData.__len__(), metric, disMetric, norm)

    for k in range(0, 1):
        teachingSet, testSet = createTeachingAndTestSets(patients)
        for w in range(0, 1):
            for metric in algParameters.getMetrics():
                for disMetric in algParameters.getDistanceMetrics():
                    for norm in algParameters.getNormalization():
                        for i in range(0, ksData.__len__()):
                            features = []
                            for param in range(i+1):
                                features.append(ksData[param].getParamID())
                            calulateResultForAlgorithms(teachingSet, testSet, features, algorithms, halo[metric+disMetric+str(norm)], i)
            print(str(k) + "-" + str(w))
            var = testSet
            testSet = teachingSet
            teachingSet = var

    for metric in algParameters.getMetrics():
        for disMetric in algParameters.getDistanceMetrics():
            for norm in algParameters.getNormalization():
                saveDataToFile(halo[metric+disMetric+str(norm)])


def calulateResultForAlgorithms(teachingSet, testSet, features, algorithms, classificationResults, amountOfFeatures):
    norm = classificationResults.normalization
    metric = classificationResults.metric
    disMetric = classificationResults.distanceMetric
    for algorithm in algorithms:
        alg_name = str(algorithm.getKValue()) + algorithm.getType()
        if algorithm.getType() == 'nm_alg':
            score, matrix = eval(algorithm.getType())(teachingSet, testSet, features, disMetric, norm, metric)
            classificationResults.getAlgorithmsResults()[alg_name][amountOfFeatures].append(score)
        else:
            score, matrix = eval(algorithm.getType())(teachingSet, testSet, features, algorithm.getKValue(), disMetric, norm, metric)
            classificationResults.getAlgorithmsResults()[alg_name][amountOfFeatures].append(score)
        # if (classificationResults.getConfusionMatrix() == None):
        #     classificationResults.setConfusionMatrix(matrix)

def saveDataToFile(classificationResults):
    with open("results/results.csv", mode='a+', newline='') as result_file:
        result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for algorithm in classificationResults.getAlgorithmsResults():
            elements = []
            rowWithParameters = algorithm + "_" + classificationResults.getMetric() + "_" + classificationResults.getDistanceMetric() + str(classificationResults.getNormalization())
            result_writer.writerow([rowWithParameters])
            for feature in classificationResults.getAlgorithmsResults()[algorithm]:
                avarageOfArray = mean(classificationResults.getAlgorithmsResults()[algorithm][feature])
                elements.append(avarageOfArray)
                result_writer.writerow([feature, "{0:.3f}".format(avarageOfArray), "{0:.3f}".format(std(classificationResults.getAlgorithmsResults()[algorithm][feature]))])
                avarageAndDeviation = str("{0:.3f}".format(mean(elements))) + "(" + str("{0:.3f}".format(std(elements))) + ")"
            result_writer.writerow(["Average(std): ", avarageAndDeviation])


#   dataSet - {B/M: { number of parameter: [] }
    # odchylenie
    # inna metryka

def main():
    file_name = "../resources/wdbc.data"
    size_of_data = 32
    amountOfFeatures = 30
    patients = file_reader.loadDataFromFile(file_name, size_of_data)
    teachingSet, testSet = createTeachingAndTestSets(patients)
    algorithms = [Alghorithm('NM', 'nm_alg'), Alghorithm('NN_1', 'knn_alg',  1), Alghorithm('NN_5','knn_alg', 5), Alghorithm('NN_9','knn_alg', 9)]
    dataSet = createDataStructure(patients, amountOfFeatures)

    algParameters = AlgParameters()
    ksData = kolmogorovTest(dataSet, amountOfFeatures)
    # createReasearches(patients, algorithms, ksData, algParameters)

    # creating confusion matrix
    accuracy_nm_alg, matrix = nm_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 26)],
                                     algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1],
                                     algParameters.getMetrics()[0])
    print("26 cech, manhatan, bez normalizacji, accuracy, NM", accuracy_nm_alg, "\n" ,matrix)
    accuracy_knn_alg1, matrix1 = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 26)], 1,
                                       algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1],
                                       algParameters.getMetrics()[0])
    print("26 cech, manhatan, bez normalizacji, accuracy, 1-NN", accuracy_knn_alg1, "\n" , matrix1)
    accuracy_knn_alg2, matrix2 = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 26)], 5,
                                       algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1],
                                       algParameters.getMetrics()[0])
    print("26 cech, manhatan, bez normalizacji, accuracy, 5-NN", accuracy_knn_alg2,"\n", matrix2)
    accuracy_knn_alg3, matrix3 = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 26)], 9,
                                       algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1],
                                       algParameters.getMetrics()[0])
    print("26 cech, manhatan, bez normalizacji, accuracy, 9-NN", accuracy_knn_alg3, "\n", matrix3)
    accuracy_knn_alg4, matrix4 = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 26)], 9,
                                       algParameters.getDistanceMetrics()[0], algParameters.getNormalization()[1],
                                       algParameters.getMetrics()[1])
    print("26 cech, euklides, bez normalizacji, balanced, 9-NN", accuracy_knn_alg4,"\n" , matrix4)
    accuracy_knn_alg5, matrix5 = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 26)], 9,
                                       algParameters.getDistanceMetrics()[0], algParameters.getNormalization()[0],
                                       algParameters.getMetrics()[1])
    print("26 cech, euklides, z normalizacja, balanced, 9-NN", accuracy_knn_alg5, "\n" ,matrix5)

    #others calculations

    # for asdf in ksData:
    #         print("{} ({}, {})".format(asdf.getParamID(), asdf.geStatistic(), asdf.getPValue()))

    # features = [7, 17, 4, 5, 12]
    # [ksData[i].getParamID() for i in range(0, 5)]
    # accuracy_knn_alg, matrix = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], 1, algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[0])
    # accuracy_knn_alg1, matrix1 = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], 1, algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[1])

    # accuracy_nm_alg, matrix = nm_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[0])

    # print(matrix)
    # print(accuracy_knn_alg)
    # print(accuracy_knn_alg1)
    # print(accuracy_nm_alg1)
    # print(accuracy_nm_alg2)
    # print(accuracy_nm_alg3)



main()
