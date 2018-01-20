#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#      300 faces In-the-wild challenge: Database and results.
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.

import dlib
import cv2

class FacialLandmarkDetector:

    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        self.dets = None
        self.shape = None
    #Detects frontal face on image
    #Returns dlib.rectangle object
    #agrument draw decides if rectangle is drawn on image
    def detect_frontal_face(self, draw=False):
        detector = dlib.get_frontal_face_detector()
        self.dets = detector(self.img, 1)
        for k, d in enumerate(self.dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            if draw==True:
                cv2.rectangle(self.img,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),2)
        return d
    #shows image inside a windows
    def showImage(self):
        cv2.imshow('image',self.img)
        cv2.waitKey(0)
    #detects facial landmarks based
    #returns list of tuples of (x,y) which represent 68 landmark points
    def detectFacialLandmarks(self, draw):
        #shape_predictor_68_face_landmarks.dat can be downloaded from
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.parts = []
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
        d = self.detect_frontal_face()
        self.shape = predictor(self.img, d)

        for i in range(self.shape.num_parts):
            self.parts.append((self.shape.part(i).x,self.shape.part(i).y))
            if draw==True:
                cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        return self.parts
    def getFacialLandmarksOfFacePart(self, faceParts, draw=False):
        if self.shape == None:
            self.parts = self.detectFacialLandmarks(False)
        foundParts = []
        if "Mouth" in faceParts:
            for i in range(48, 68):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "RightEyebrow" in faceParts:
            for i in range(17, 22):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "LeftEyebrow" in faceParts:
            for i in range(22, 27):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "RightEye" in faceParts:
            for i in range(36, 42):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "LeftEye" in faceParts:
            for i in range(42, 48):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "Nose" in faceParts:
            for i in range(27, 36):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        if "Jaw" in faceParts:
            for i in range(0, 17):
                foundParts.append(self.parts[i])
                if draw==True:
                    cv2.circle(self.img,(self.shape.part(i).x,self.shape.part(i).y), 2, (0,0,255), -1)
        return foundParts
    def extractFacePart(self, facePart, destinationFolder):
        if facePart == "EyeRegion":
            parts = self.getFacialLandmarksOfFacePart(["RightEye", "LeftEye", "RightEyebrow", "LeftEyebrow"])
            top, left, bottom, right = maxRectangle(parts)
            region = self.img[top:bottom+10, left:right]
        return region

def maxRectangle(parts):
    maxTop = 1111111111
    maxLeft = 111111111
    maxBottom = -100
    maxRight = -100
    for part in parts:
        if part[0] < maxLeft:
            maxLeft = part[0]
        if part[0] > maxRight:
            maxRight = part[0]
        if part[1] < maxTop:
            maxTop = part[1]
        if part[1] > maxBottom:
            maxBottom = part[1]
    return maxTop, maxLeft, maxBottom, maxRight

if __name__ == "__main__":
    image_name = "/home/matej/FER_current/Projekt/Project_Deidentification_Kazemi/baza_XMVTS2/000/000_1_1.ppm"
    detector = FacialLandmarkDetector(image_name)
    #detector.detect_frontal_face(True)
    #detector.showImage()
    #parts = detector.detectFacialLandmarks(True)
    #detector.showImage()
    #print(parts)
    #foundParts = detector.getFacialLandmarksOfFacePart(["Nose", "Mouth"], True)
    #detector.showImage()
    #print(foundParts)
    ROI = detector.extractFacePart("EyeRegion", "bezz")
    cv2.imshow('image',ROI)
    cv2.waitKey(0)
