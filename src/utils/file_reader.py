from src.models.patient import Patient


def loadDataFromFile(file_name, size_of_data):
    patients = []
    data = readDataFromFile(file_name)
    data_array = data.split("\n")
    for row in data_array:
        formatted_row = row.split(",")
        if formatted_row.__len__() == size_of_data:
            patient = Patient(formatted_row[0], formatted_row[1], formatted_row[2:size_of_data])
            patients.append(patient)
    return patients


def readDataFromFile(file_name):
    file = open(file_name, "r")
    return file.read()
