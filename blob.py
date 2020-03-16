import cv2 as cv
import numpy as np
import math as m


class Blob:
    # Area of a BLOB is the number of pixels the BLOB consists of.


    def __init__(self):
        self.Eccentricity = 0
        self.Ratio1 = 0  # MinorAxisLength / MajorAxisLength
        self.Ratio2 = 0  # Perimeter / Area

    def extractfeature(self, image):
        res = image[:]
        ret, thresh = cv.threshold(res, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE )  #(thresh, 1, 2)
        cnt = contours[0]
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        self.Ratio2 = perimeter/area

        ellipse = cv.fitEllipse(cnt)
        (x, y), (Major, minor), angle = ellipse
        MA = max(Major, minor)
        ma = min(Major, minor)
        self.Ratio1 = MA / ma
        a = MA/2
        b = ma/2
        c = m.sqrt(m.pow(a,2) - m.pow(b,2))
        e = c/a
        self.Eccentricity = e

        return [self.Eccentricity, self.Ratio1, self.Ratio2]

    def features(self):
        return [self.Eccentricity, self.Ratio1, self.Ratio2]

    def findobjects(self, image, features):
        feat = np.array(features, float)
        res = image[:]
        ret, thresh = cv.threshold(res, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # (thresh, 1, 2)
        labels = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            perimeter = cv.arcLength(cnt, True)
            self.Ratio2 = perimeter / area

            ellipse = cv.fitEllipse(cnt)
            (x, y), (Major, minor), angle = ellipse
            MA = max(Major, minor)
            ma = min(Major, minor)
            self.Ratio1 = MA / ma

            a = MA / 2
            b = ma / 2
            c = m.sqrt(m.pow(a, 2) - m.pow(b, 2))
            e = c / a
            self.Eccentricity = e

            distancelist = []
            label = []
            for i in feat:
                data = m.sqrt(m.pow(i[0] - self.Eccentricity, 2) + m.pow(i[1] - self.Ratio1, 2) + m.pow(i[2] - self.Ratio1, 2))
                distancelist.append(data)
                label.append(i[3])
            index = distancelist.index(min(distancelist))
            labels.append(label[index])
            distancelist.clear()
        return labels
    # WHAT IS BLOB
    # A Blob is a group of connected pixels in an image that share some common property ( E.g grayscale value ).
    # LINK: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    #       http://what-when-how.com/introduction-to-video-and-image-processing/blob-analysis-introduction-to-video-and-image-processing-part-1/
    #       https://www.mathsisfun.com/geometry/eccentricity.html
    #       https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=connectedcomponents#connectedcomponents
    #       https://www.tutorialspoint.com/python3/python_classes_objects.htm
    #       https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    #       https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    #       https://stackoverflow.com/questions/39486869/how-to-fit-an-ellipse-contour-with-4-points

    # data = cv.connectedComponentsWithStats(image,connectivity=4)
    # data[number of components in image including background as 0, labeledImageArray, stats, centroids]
    # stats sequence: cv.CC_STAT_LEFT, cv.CC_STAT_TOP, cv.CC_STAT_WIDTH, cv.CC_STAT_HEIGHT, cv.CC_STAT_AREA, cv.CC_STAT_MAX
    # temp = data[2]
    # Blob.area = self.__Area = data[2][1][4]
    # return data
    # [1, 1, 4, 4, 2, 1, 1, 4, 2, 1, 1, 1, 1, 1]