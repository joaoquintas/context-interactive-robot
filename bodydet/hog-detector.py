import sys
import os
import cv2
import numpy as np

from utils import read_images

if __name__ == "__main__":

    # script parameters
    if len(sys.argv) < 2:
        raise Exception("No image given.")
    inFileName = sys.argv[1]
    outFileName = None
    if len(sys.argv) > 2:
        outFileName = sys.argv[2]
    if outFileName == inFileName:
        outFileName = None


    # [X, y] = read_images(sys.argv[1],sz)
    [Names, X, y] = read_images("../data/people/INRIAPerson/Test/pos",sz=None)
    imgBGR = []

    # detection begins here
    # img = cv2.imread(inFileName)
    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/person_007.png")
    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/person_011.png")
    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/crop001633.png")

    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/crop001638.png")

    # img = cv2.imread("../data/people/Cam40small1.png")
    # img = cv2.imread("../data/people/cam51.png")
    # imgOut = cv2.resize(img, (640,480))
    # imgOut = cv2.resize(img, (128,68))

    # imgOut = img
    # imgOut = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

    # set up detectors # initialize the HOG descriptor/person detector

    # winSize = (64,128)
    # blockSize = (16,16)
    # blockStride = (8,8)
    # cellSize = (8,8)
    # nbins = 9
    # # derivAperture = 1
    # winSigma = -1
    # # histogramNormType = 0
    # L2HysThreshold = 2.0000000000000001e-01
    # gammaCorrection = True
    # nlevels = 64
    # hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,winSigma,L2HysThreshold,gammaCorrection,nlevels)
    # # hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
    # #                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    # # hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)


    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    # winStride = (8,8)
    # padding = (8,8)
    # locations = ((10,20),)
    # hist = hog.compute(img,winStride,padding,locations)


    #for all images in the list X
    for imgName, imgOut in zip(Names,X):

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(imgOut, winStride=(4, 4), padding=(8, 8), scale=1.2)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            imgBGR = cv2.cvtColor(imgOut,cv2.COLOR_GRAY2BGR)
            cv2.rectangle(imgBGR, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        # for (xA, yA, xB, yB) in pick:
        #    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show the output images
        cv2.imshow("People detected: " + imgName, imgBGR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # imgBinary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret, imgBinary = cv2.threshold(imgBinary, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Binary image", imgBinary)
    # cv2.waitKey(0)