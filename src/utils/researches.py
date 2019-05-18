from functools import reduce
import random
from src.models.classification_result import ClassificationResult
from src.algorithms import nm_alg, knn_alg
from src.utils.csv_file_writer import saveDataToFile

# parameters to cross validation
amountOfLoops = 5

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


def createResearches(patients, algorithms,ksData ,algParameters):

    tmp = {}
    for metric in algParameters.getMetrics():
        for disMetric in algParameters.getDistanceMetrics():
            for norm in algParameters.getNormalization():
                tmp[metric+disMetric+str(norm)] = ClassificationResult(ksData.__len__(), metric, disMetric, norm)

    for k in range(0, amountOfLoops):
        teachingSet, testSet = createTeachingAndTestSets(patients)
        for w in range(0, 2):
            for metric in algParameters.getMetrics():
                for disMetric in algParameters.getDistanceMetrics():
                    for norm in algParameters.getNormalization():
                        for i in range(0, ksData.__len__()):
                            features = []
                            for param in range(i+1):
                                features.append(ksData[param].getParamID())
                            calulateResultForAlgorithms(teachingSet, testSet, features, algorithms, tmp[metric+disMetric+str(norm)], i)
            var = testSet
            testSet = teachingSet
            teachingSet = var

    for metric in algParameters.getMetrics():
        for disMetric in algParameters.getDistanceMetrics():
            for norm in algParameters.getNormalization():
                saveDataToFile(tmp[metric+disMetric+str(norm)])


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
