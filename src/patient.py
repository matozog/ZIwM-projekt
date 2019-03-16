
class Patient:
    id = 0
    cancer_type = ''
    values = []

    def __init__(self, id, type_cancer, values):
        self.id = id
        self.type_cancer = type_cancer
        self.values = values

    def getID(self):
        return self.id

    def getCancerType(self):
        return self.type_cancer

    def getInputValues(self):
        return self.values