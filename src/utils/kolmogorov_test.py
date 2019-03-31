from src.models.ksType import ksType
from scipy.stats import ks_2samp


def kolmogorovTest(dataSet, amountOfFeatures):
    ksData = []
    for feature in range(0, amountOfFeatures):
        statistic, pvalue = ks_2samp(dataSet["M"][feature], dataSet["B"][feature])
        if pvalue < 0.002:
            ksData.append(ksType(feature, statistic, pvalue))

    ksData.sort(key=lambda val: val.geStatistic(), reverse=True)

    return ksData
