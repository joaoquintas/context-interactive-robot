import sys, os, csv
import cv2
import numpy as np
from matplotlib import pyplot

# import for measure execution time of small code snippets (https://docs.python.org/3.4/library/timeit.html)
import timeit

# import for utility method to read images
from .utils import non_max_suppression_fast, read_images, parse_pascal


class Detector:
    def detect(self, src):
        raise NotImplementedError("Every Detector must implement the detect method.")


class SkinDetector(Detector):
    """
    Implements common color thresholding rules for the RGB, YCrCb and HSV color
    space. The values are taken from a paper, which I can't find right now, so
    be careful with this detector.

    """

    def _R1(self, BGR):
        # channels
        B = BGR[:, :, 0]
        G = BGR[:, :, 1]
        R = BGR[:, :, 2]
        e1 = (R > 95) & (G > 40) & (B > 20) & (
        (np.maximum(R, np.maximum(G, B)) - np.minimum(R, np.minimum(G, B))) > 15) & (np.abs(R - G) > 15) & (R > G) & (
             R > B)
        e2 = (R > 220) & (G > 210) & (B > 170) & (abs(R - G) <= 15) & (R > B) & (G > B)
        return (e1 | e2)

    def _R2(self, YCrCb):
        Y = YCrCb[:, :, 0]
        Cr = YCrCb[:, :, 1]
        Cb = YCrCb[:, :, 2]
        e1 = Cr <= (1.5862 * Cb + 20)
        e2 = Cr >= (0.3448 * Cb + 76.2069)
        e3 = Cr >= (-4.5652 * Cb + 234.5652)
        e4 = Cr <= (-1.15 * Cb + 301.75)
        e5 = Cr <= (-2.2857 * Cb + 432.85)
        return e1 & e2 & e3 & e4 & e5

    def _R3(self, HSV):
        H = HSV[:, :, 0]
        S = HSV[:, :, 1]
        V = HSV[:, :, 2]
        return ((H < 25) | (H > 230))

    def detect(self, src):
        if np.ndim(src) < 3:
            return np.ones(src.shape, dtype=np.uint8)
        if src.dtype != np.uint8:
            return np.ones(src.shape, dtype=np.uint8)
        srcYCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
        srcHSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        skinPixels = self._R1(src) & self._R2(srcYCrCb) & self._R3(srcHSV)
        return np.asarray(skinPixels, dtype=np.uint8)


class CascadedDetector(Detector):
    """
    Uses the OpenCV cascades to perform the detection. Returns the Regions of Interest, where
    the detector assumes a face. You probably have to play around with the scaleFactor,
    minNeighbors and minSize parameters to get good results for your use case. From my
    personal experience, all I can say is: there's no parameter combination which *just
    works*.
    """

    def __init__(self, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scaleFactor=1.2, minNeighbors=5,
                 minSize=(30, 30)):
        if not os.path.exists(cascade_fn):
            raise IOError("No valid cascade found for path=%s." % cascade_fn)
        self.cascade = cv2.CascadeClassifier(cascade_fn)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def detect(self, src):
        if np.ndim(src) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src = cv2.equalizeHist(src)
        rects = self.cascade.detectMultiScale(src, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors,
                                              minSize=self.minSize)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects


class SkinFaceDetector(Detector):
    """
    Uses the SkinDetector to accept only faces over a given skin color tone threshold (ignored for
    grayscale images). Be careful with skin color tone thresholding, as it won't work in uncontrolled
    scenarios (without preprocessing)!

    """

    def __init__(self, threshold=0.3, cascade_fn="./cascades/haarcascade_frontalface_alt2.xml", scaleFactor=1.2,
                 minNeighbors=5, minSize=(30, 30)):
        self.faceDetector = CascadedDetector(cascade_fn=cascade_fn, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                             minSize=minSize)
        self.skinDetector = SkinDetector()
        self.threshold = threshold

    def detect(self, src):
        rects = []
        for i, r in enumerate(self.faceDetector.detect(src)):
            x0, y0, x1, y1 = r
            face = src[y0:y1, x0:x1]
            skinPixels = self.skinDetector.detect(face)
            skinPercentage = float(np.sum(skinPixels)) / skinPixels.size
            print(skinPercentage)
            if skinPercentage > self.threshold:
                rects.append(r)
        return rects


class BodyDetector(Detector):
    """
    Uses the OpenCV cascades to perform the detection. Returns the Regions of Interest, where
    the detector assumes a face. You probably have to play around with the scaleFactor,
    minNeighbors and minSize parameters to get good results for your use case. From my
    personal experience, all I can say is: there's no parameter combination which *just
    works*.
    """

    def __init__(self, cascade_fn="/Users/Utilizador/opencv-3.0.0/data/haarcascades/haarcascade_fullbody.xml",
                 scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)):
        if not os.path.exists(cascade_fn):
            raise IOError("No valid cascade found for path=%s." % cascade_fn)
        self.cascade = cv2.CascadeClassifier(cascade_fn)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def detect(self, src):
        if np.ndim(src) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src = cv2.equalizeHist(src)
        rects = self.cascade.detectMultiScale(src, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors,
                                              minSize=self.minSize)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects


def write_img_hist():
    directory = "out/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    imgList = {
        # Haar fails
        # "person_263.png"         ,"person_350.png"         ,"person_and_bike_164.png",
        "person_085.png", "person_055.png", "person_212.png",
        # "person_and_bike_093.png","person_and_bike_105.png","person_and_bike_085.png",
        # "person_and_bike_163.png","person_and_bike_153.png","person_107.png","person_009.png",
        # "crop_000013.png"        ,"person_and_bike_107.png","person_and_bike_046.png",
        # "person_and_bike_096.png","crop_000017.png"        ,"person_and_bike_047.png",
        # "person_and_bike_103.png","person_and_bike_095.png","person_and_bike_098.png",
        # "person_and_bike_165.png","person_and_bike_091.png","person_059.png",
        # "person_246.png"         ,"person_and_bike_089.png","person_and_bike_106.png",
        # "crop_000008.png"        ,"person_and_bike_064.png","person_and_bike_088.png",
        # "person_and_bike_094.png","person_and_bike_087.png","person_and_bike_061.png",

        # HOG fails
        # "crop_000001.png"        ,"crop_000015.png"        ,"person_and_bike_080.png",
        # "person_and_bike_073.png","person_and_bike_092.png","crop001722.png",
        # "person_and_bike_162.png","person_and_bike_114.png","person_and_bike_086.png",
        # "person_and_bike_161.png","person_033.png"         ,"person_and_bike_004.png",
        # "person_029.png"         ,"person_and_bike_177.png","person_and_bike_101.png",
        # "person_and_bike_099.png","person_and_bike_083.png","person_030.png",
        # "crop_000007.png"        ,"person_and_bike_159.png","person_065.png",
        # "person_115.png"         ,"crop001634.png"
    }

    if imgName in imgList:
        cv2.imwrite("out/out_" + imgName, imgBGR)
        # imgHist = cv2.cvtColor(imgOut,cv2.COLOR_BGR2GRAY)
        # calc histrogram of image
        hist = cv2.calcHist([imgOut], [0], None, [256], [0, 256])
        pyplot.hist(imgOut.ravel(), 256, [0, 256])
        hist_filename = directory + "out_hist_" + imgName
        pyplot.savefig(hist_filename)
    # pyplot.show(block=False)


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

    # detection begins here
    # img = cv2.imread(inFileName)
    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/person_007.png")
    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/person_011.png")
    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/crop001633.png")

    # img = cv2.imread("../data/people/INRIAPerson/Test/pos/crop001638.png")

    # img = cv2.imread("../data/people/Cam40small1.png")
    # img = cv2.imread("../data/people/cam51.png")

    # imgOut = img

    # imgOut = cv2.resize(img, (128, 68))
    # imgOut = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)



    ######################################
    # set up detectors
    #
    #
    detector = BodyDetector(cascade_fn="../data/people/body10/haarcascade_fullbody.xml", scaleFactor=1.05,
                            minNeighbors=1)
    # detector = BodyDetector(cascade_fn="../data/people/body10/haarcascade_upperbody.xml")
    # detector = BodyDetector(cascade_fn="../data/people/body10/haarcascade_lowerbody.xml")

    # detector = BodyDetector(cascade_fn="/Users/Utilizador/opencv-3.0.0/data/haarcascades/haarcascade_fullbody.xml", scaleFactor=1.05,
    # 						minNeighbors=1)
    # detector = BodyDetector(cascade_fn="/Users/Utilizador/opencv-3.0.0/data/haarcascades/haarcascade_upperbody.xml")
    # detector = BodyDetector(cascade_fn="/Users/Utilizador/opencv-3.0.0/data/haarcascades/haarcascade_lowerbody.xml")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # rect = detector.detect(imgOut)



    ######################################
    # Reading images
    #
    #
    print("Reading images...")

    [Names, X, y] = read_images("../data/people/INRIAPerson/Test/pos", sz=None)
    # [Names, X, y] = sorted(([Names, X, y]), key=lambda image:X[0])
    imgBGR = []

    # for all images in list X
    for imgIdx, imgName, imgOut in zip(y, Names, X):

        # parse PASCAL Annotation files
        (imSize, n_persons, bounding_boxes) = parse_pascal(imgName)
        n_persons = int(n_persons)

        dist_threshold = 0.15 * (int(imSize[0]) + int(imSize[1]))
        # print(dist_threshold)

        imgBGR = cv2.cvtColor(imgOut, cv2.COLOR_GRAY2BGR)

        # draw annotation in the image
        for bb in range(0, len(bounding_boxes)):
            cv2.rectangle(imgBGR, (int(bounding_boxes[bb][0]), int(bounding_boxes[bb][1])),
                          (int(bounding_boxes[bb][2]), int(bounding_boxes[bb][3])), (255, 0, 0), 1)
            cv2.putText(imgBGR, 'annotation', (int(bounding_boxes[bb][0]), int(bounding_boxes[bb][1])),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)




        ######################################
        # Haar detection
        # compare results (count hit, miss, error)
        # hit when bounding boxes are correct (true positive)
        # error when bounding boxes not correct (false positive)
        print("Haar detection...")

        hit_haar_list = [];

        # Haar detection of people
        t_haar = timeit.timeit(stmt='detector.detect(imgOut)', setup='from __main__ import detector, imgOut', number=1)
        # print("haar detection: {}s".format(t_haar))

        # calculate if detection is a hit or error (compare with all bounding boxes condition positive)
        for i, r in enumerate(detector.detect(imgOut)):
            x0, y0, x1, y1 = r
            for bb in range(0, len(bounding_boxes)):
                dist_haar = np.sqrt(
                        np.square(int(bounding_boxes[bb][0]) - x0) + np.square(int(bounding_boxes[bb][1]) - y0)) + \
                            np.sqrt(
                                    np.square(int(bounding_boxes[bb][2]) - x1) + np.square(int(bounding_boxes[bb][3]) - y1))
                # print("dist_haar: {}".format(dist_haar))
                if dist_haar < dist_threshold:
                    hit_haar_list.append(bb)
                    break

            # represent detection in image
            cv2.rectangle(imgBGR, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.putText(imgBGR, 'haar-detection', (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        # remove duplicates from hit_haar and error_haar
        hit_haar_list = set(hit_haar_list)

        # initialize variables for number of detection, hits, errors, misses
        n_det_haar = 0; hit_haar = 0; error_haar = 0; miss_haar = 0

        # calculate number of detections
        n_det_haar = len(detector.detect(imgOut))

        # calculate hits true positives
        hit_haar = len(hit_haar_list)

        # calculate error false positive
        error_haar = n_det_haar - hit_haar

        # calculate misses (false negatives)
        miss_haar = n_persons - hit_haar

        # print("\nn_persons:{}; n_det_haar:{}; hit_haar:{}; error_haar:{}; miss_haar:{};".format(n_persons, n_det_haar,
        #                                                                                         hit_haar,
        #                                                                                         error_haar,
        #                                                                                         miss_haar))



        ######################################
        # HOG detection of people in the image
        # compare results (count hit, miss, error)
        # hit when bounding boxes are correct (true positive)
        # error when bounding boxes not correct (false positive)
        print("Hog detection...")

        hit_hog_list = [];

        t_hog = timeit.timeit(stmt='hog.detectMultiScale(imgOut, winStride=(4, 4), padding=(8, 8), scale=1.2)',
                              setup='from __main__ import hog, imgOut', number=1)
        # print("hog detection: {}s".format(t_hog))

        (rects, weights) = hog.detectMultiScale(imgOut, winStride=(4, 4), padding=(16, 16), scale=1.1)
        rects = non_max_suppression_fast(rects, 0.2)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:

            # calculate if detection is a hit or error
            for bb in range(0, len(bounding_boxes)):
                dist_hog = np.sqrt(np.square(int(bounding_boxes[bb][0]) - x) + np.square(int(bounding_boxes[bb][1]) - y)) + \
                           np.sqrt(np.square(int(bounding_boxes[bb][2]) - (x + w)) + np.square(
                               int(bounding_boxes[bb][3]) - (y + h)))
                # print("dist_hog: {}".format(dist_hog))
                if dist_hog < dist_threshold:
                    hit_hog_list.append(bb)
                    break

            # represent detection in image
            cv2.rectangle(imgBGR, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(imgBGR, 'hog-detection', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        # remove duplicates from hit_haar and error_haar
        hit_hog_list = set(hit_hog_list)

        # initialize variables for number of detection, hits, errors, misses
        n_det_hog = 0; hit_hog = 0; error_hog = 0; miss_hog = 0

        # calculates number of detections
        n_det_hog = len(rects)

        # calculate hits true positives
        hit_hog = len(hit_hog_list)

        # calculate error false positive
        error_hog = n_det_hog - hit_hog

        # calculate misses (false negatives)
        miss_hog = n_persons - hit_hog

        # print("\nn_persons:{}; n_det_hog:{}; hit_hog:{}; error_hog:{}; miss_hog:{};".format(n_persons, n_det_hog,
        #                                                                                         hit_hog,
        #                                                                                         error_hog,
        #                                                                                         miss_hog))


        # display image
        # cv2.imshow("People detected: " + imgName, imgBGR)

        # write images and histograms to files
        # write_img_hist()



        ######################################
        # calculate precision, recall and f-measure for haar detection in current image
        #
        #
        if n_det_haar > 0:
            precision_haar = hit_haar/n_det_haar
        else:
            precision_haar = 0
        if n_persons > 0:
            recall_haar = hit_haar/n_persons
        else:
            recall_haar = 0
        if (precision_haar+recall_haar)>0:
            f_measure_haar = 2*(precision_haar*recall_haar)/(precision_haar+recall_haar)
        else:
            f_measure_haar = 0

        # calculate precision, recall and f-measure for hog detection in current image
        if n_det_hog > 0:
            precision_hog = hit_hog/n_det_hog
        else:
            precision_hog = 0
        if n_persons > 0:
            recall_hog = hit_hog/n_persons
        else:
            recall_hog = 0
        if (precision_hog+recall_hog)>0:
            f_measure_hog = 2*(precision_hog*recall_hog)/(precision_hog+recall_hog)
        else:
            f_measure_hog = 0


        # print output row for image algorithms analysis
        print("\n{},{},{}|{:.4f},{},{},{},{},{:.2f},{:.2f},{:.2f}|{:.4f},{},{},{},{},{:.2f},{:.2f},{:.2f}".format(
                imgIdx, imgName, n_persons,
                t_haar, n_det_haar, hit_haar, error_haar, miss_haar,precision_haar, recall_haar, f_measure_haar,
                t_hog, n_det_hog, hit_hog, error_hog, miss_hog, precision_hog, recall_hog, f_measure_hog))




        ######################################
        # write row to file
        #
        #

        # prepare file to write results to file
        with open('results_haar_hog_prec_rec_f_2.csv', 'a', newline='') as f:
        	writer = csv.writer(f)
        	row = [imgIdx, imgName, n_persons,
                   t_haar, n_det_haar, hit_haar, error_haar, miss_haar,precision_haar, recall_haar, f_measure_haar,
                   t_hog, n_det_hog, hit_hog, error_hog, miss_hog, precision_hog, recall_hog, f_measure_hog]
        	writer.writerow(row)
        f.close()


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
