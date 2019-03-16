from src.utils import file_reader
from src.patient import Patient


def loadDataFromFile(file_name, size_of_data):
    patients = []
    data = file_reader.readDataFromFile(file_name)
    data_array = data.split("\n")
    for row in data_array:
        formatted_row = row.split(",")
        if formatted_row.__len__() == size_of_data:
            patient = Patient(formatted_row[0], formatted_row[1], formatted_row[2:size_of_data-1])
            patients.append(patient)
    return patients


def main():
    file_name = "../resources/wdbc.data"
    size_of_data = 32
    patients = loadDataFromFile(file_name, size_of_data)
    for patient in patients:
        if patient.getCancerType() == "M":
            print(patient.getInputValues()[0])


main()
