from numpy import std, mean, asarray
import csv

fileToSave = "results/results.csv"


def saveDataToFile(classificationResults):
    with open(fileToSave, mode='a+', newline='') as result_file:
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