
class AlgParameters:
    __metrics = ["accuracy", "balanced_accuracy", "cohen_kappa"]
    __distanceMetrics = ['euclidean', 'manhattan']
    __normalizations = [True, False]

    def getMetrics(self):
        return self.__metrics

    def getDistanceMetrics(self):
        return self.__distanceMetrics

    def getNormalization(self):
        return self.__normalizations