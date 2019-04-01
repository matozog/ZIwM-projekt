from sklearn import neighbors
from sklearn.metrics import accuracy_score
from src.utils.data_structure import prepareDataSet


def knn_alg(teachingSet, testSet, features, kValue, distanceMetrics, normalization):
    trainDataFeatures, trainDataLabelFeatures = prepareDataSet(teachingSet, features, normalization)
    testDataFeatures, testDataLabelFeatures = prepareDataSet(testSet, features, normalization)

    classifier = neighbors.KNeighborsClassifier(n_neighbors=kValue, metric=distanceMetrics)
    classifier.fit(trainDataFeatures, trainDataLabelFeatures)
    predictions = classifier.predict(testDataFeatures)
    score = accuracy_score(testDataLabelFeatures, predictions)

    return score
