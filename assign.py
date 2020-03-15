import numpy as np
import cv2 as cv
import blob as b
import csv as cs
import math as m

# training images
arrytemp1 = cv.imread("temp1.png", 0)
arrytemp2 = cv.imread("temp2.png", 0)
arrytemp3 = cv.imread("temp3.png", 0)
arrytemp4 = cv.imread("temp4.png", 0)
# testing image
assignment = cv.imread("assignment1.png", 0)

# creating class objects
obj1 = b.Blob()
obj2 = b.Blob()
obj3 = b.Blob()
obj4 = b.Blob()

# extracting features from training images
data1 = obj1.extractfeature(arrytemp1)
data2 = obj2.extractfeature(arrytemp2)
data3 = obj3.extractfeature(arrytemp3)
data4 = obj4.extractfeature(arrytemp4)
# function returning the array of features it extracted
array1 = obj1.features()
array2 = obj2.features()
array3 = obj3.features()
array4 = obj4.features()

# writing features in csv, so that we do not need to train again
csvinputdata = [array1, array2, array3, array4]
with open('person.csv','w') as csvfile:
    writer=cs.writer(csvfile)
    writer.writerows(csvinputdata)
csvfile.close()

# reading from csv
csvoutputdata = []
with open('person.csv', 'r') as csvFile:
    reader = cs.reader(csvFile)
    for row in reader:
        csvoutputdata.append(row)
csvFile.close()
print(csvoutputdata)

res = b.Blob()
resdata = res.findobjects(assignment,csvoutputdata)
print(resdata)