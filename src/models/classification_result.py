
class ClassificationResult:


    def __init__(self, amountOfFeatures, metric, distanceMetric, normalization):
        self.accuracy = 0
        self.deviation = 0
        self.usingAlgorithm = ""
        self.confusionMatrix = None
        self.metric = metric
        self.distanceMetric = distanceMetric
        self.normalization = normalization
        self.__algorithmResults = {}
        self.__algorithmResults["0nm_alg"] = {}
        self.__algorithmResults["1knn_alg"] = {}
        self.__algorithmResults["5knn_alg"] = {}
        self.__algorithmResults["9knn_alg"] = {}
        for x in range(0, amountOfFeatures):
            self.__algorithmResults["0nm_alg"][x] = []
            self.__algorithmResults["1knn_alg"][x] = []
            self.__algorithmResults["5knn_alg"][x] = []
            self.__algorithmResults["9knn_alg"][x] = []

    def getMetric(self):
        return self.metric

    def getDistanceMetric(self):
        return self.distanceMetric

    def getNormalization(self):
        return self.normalization

    def getConfusionMatrix(self):
        return self.confusionMatrix

    def setConfusionMatrix(self, matrix):
        self.confusionMatrix = matrix

    def getAlgorithmsResults(self):
        return self.__algorithmResults

    def setAccuracy(self, accuracy):
        self.accuracy = accuracy

    def setDeviation(self, deviation):
        self.deviation = deviation

    def setConfusionMatrix(self, confusionMatrix):
        self.confusionMatrix = confusionMatrix

    def getAccuracy(self):
        return self.accuracy

    def getDeviation(self):
        return self.deviation
