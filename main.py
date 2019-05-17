from src.algorithms import nm_alg, knn_alg
from src.models import Alghorithm
from src.models.alghoritm_parameters import AlgParameters
from src.utils import file_reader, createDataStructure
from src.utils import kolmogorovTest
from src.utils.researches import createResearches, createTeachingAndTestSets

#   dataSet - {B/M: { number of parameter: [] }

def main():
    file_name = "../resources/wdbc.data"
    size_of_data = 32
    amountOfFeatures = 30
    patients = file_reader.loadDataFromFile(file_name, size_of_data)
    teachingSet, testSet = createTeachingAndTestSets(patients)
    algorithms = [Alghorithm('NM', 'nm_alg'), Alghorithm('NN_1', 'knn_alg',  1), Alghorithm('NN_5','knn_alg', 5),
                  Alghorithm('NN_9','knn_alg', 9)]
    dataSet = createDataStructure(patients, amountOfFeatures)

    algParameters = AlgParameters()
    ksData = kolmogorovTest(dataSet, amountOfFeatures)

    createResearches(patients, algorithms, ksData, algParameters)  # function using to create researches

    # creating example confusion matrix
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

    #others simple examples

    # features = [7, 17, 4, 5, 12]
    # accuracy_knn_alg, matrix = knn_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], 1, algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[0])
    # accuracy_nm_alg, matrix = nm_alg(teachingSet, testSet, [ksData[i].getParamID() for i in range(0, 5)], algParameters.getDistanceMetrics()[1], algParameters.getNormalization()[1], algParameters.getMetrics()[0])

    # print(accuracy_knn_alg)
    # print(accuracy_nm_alg)


main()
