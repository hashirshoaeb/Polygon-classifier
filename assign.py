import numpy as np
import cv2 as cv
import blob as b
import csv as cs
import math as m
import os

# importing training images
training_imgs = []
training_dir = './train'
img_paths = os.listdir(training_dir)
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
    features.append(f)


# writing features in csv, so that we do not need to train again
csvinputdata = features
with open('features.csv','w') as csvfile:
    writer=cs.writer(csvfile)
    writer.writerows(csvinputdata)
csvfile.close()


def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # https://thispointer.com/python-add-a-column-to-an-existing-csv-file/
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = cs.reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = cs.writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)


# Add a list as column
label_list = ['1', '3', '2', '4'] # based on image paths (into numbers)
# label_list = img_paths
add_column_in_csv('features.csv', 'features_with_labels.csv', lambda row, line_num: row.append(label_list[line_num - 1]))

# reading from csv
csvoutputdata = []
with open('features_with_labels.csv', 'r') as csvFile:
    reader = cs.reader(csvFile)
    for row in reader:
        csvoutputdata.append(row)
csvFile.close()
# print(csvoutputdata)

res = b.Blob()
resdata = res.findobjects(test,csvoutputdata)
print(resdata)