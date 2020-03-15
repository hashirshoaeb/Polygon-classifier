import numpy as np
import cv2 as cv
import blob as b
import csv as cs
import math as m
import os

# importing training images
training_imgs = []
training_dir = './train'
img_paths =  os.listdir(training_dir)
for img_path in img_paths:
    t = cv.imread(training_dir + '/' + img_path, 0)
    training_imgs.append(t)

# importing testing image
test = cv.imread("test/test.png", 0)

# extracting features
features = []
for img in training_imgs:
    # creating class objects
    obj = b.Blob()
    # extracting features from training images
    obj.extractfeature(img)
    # function returning the array of features it extracted
    f = obj.features()
    print(f)
    features.append(f)


# writing features in csv, so that we do not need to train again
csvinputdata = features
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
resdata = res.findobjects(test,csvoutputdata)
print(resdata)