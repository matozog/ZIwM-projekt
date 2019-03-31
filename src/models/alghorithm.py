class Alghorithm:
    name = ''
    kValue = 0
    type = 'knn'
    accuracy = 0

    def __init__(self, name, type='knn_alg', kValue=0):
        self.kValue = kValue
        self.name = name
        self.type = type

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def setAccuracy(self, accuracy):
        self.accuracy = accuracy

    def getAccuracy(self):
        return self.accuracy

    def getKValue(self):
        return self.kValue
