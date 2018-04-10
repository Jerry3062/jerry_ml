import csv

mylist = ["123", '234', '456']
mydict = {"one":"two","three":"fout"}


def createListCSV(dataList):
    with open("demo.csv", "w") as csvFile:
        csvWriter = csv.writer(csvFile)
        print(dir(csvWriter))
        for data in dataList.values():
            csvWriter.writerow([data])
        csvFile.flush()
        csvFile.close()


createListCSV(mydict)
