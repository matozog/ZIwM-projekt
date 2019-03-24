from src.ksType import ksType
from src.utils import file_reader
from scipy.stats import ks_2samp
import random
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

    for asdf in ksData:
        print("{} ({}, {})".format(asdf.getParamID(), asdf.geStatistic(), asdf.getPValue()))

main()
