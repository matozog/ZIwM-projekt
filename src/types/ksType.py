class ksType:
    __statistic = 0
    __pvalue = 0
    __paramID = 0

    def __init__(self, paramID, statistic, pvalue):
        self.__statistic = statistic
        self.__pvalue = pvalue
        self.__paramID = paramID

    def geStatistic(self):
        return self.__statistic

    def getPValue(self):
        return self.__pvalue

    def getParamID(self):
        return self.__paramID
