import sys, os, fnmatch, datetime
import cv2
import numpy as np

# Malisiewicz et al.
# url: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    Name, X, y = [], [], []
    for dirname, dirnames, filenames in os.walk(path):

        subject_path = path
        images_list = set(fnmatch.filter(os.listdir(subject_path),'*.png'))

        try:
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                # images_list = set(fnmatch.filter(os.listdir(subject_path),'*.pgm')).difference(fnmatch.filter(os.listdir(subject_path),'*Ambient.pgm'))
                images_list = set(fnmatch.filter(os.listdir(subject_path),'*.png'))
        except:
            images_list = set(fnmatch.filter(os.listdir(path),'*.png'))

        for filename in images_list:#fnmatch.filter(os.listdir(subject_path),'*.pgm'):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, (0, 0), fx=1, fy=1)
                    Name.append(filename)
                    X.append(im)
                    y.append(c)
                except IOError as error:
                    print ("I/O error({0}): {1}".format(error.errno, error.strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
                c = c + 1
    return [Name, X, y]


def parse_pascal(imgName):

    # parse PASCAL Annotation files

    # test file
    # annotation_file = ("../data/people/INRIAPerson/Test/annotations/crop001501.txt")

    #consider revision to generalize the filename
    annotation_file = ("../data/people/INRIAPerson/Test/annotations/"+imgName).replace(".png",".txt")
    content = open(annotation_file, encoding="latin-1").read().split('\n') # use encoding="latin-1" to avoid error UnicodeDecodeError: 'utf-8' codec can't decode byte <code> in position <pos>: invalid continuation byte

    # remove spaces and comments from list
    content = [str for str in content if str != '']
    content = [str for str in content if not str.startswith('#')]

    # analyse lines of header
    for str in content:
        # header lines
        if str.startswith('Image filename : '):
            image_filename = str.split(" ")[3].replace("\"","")
        if str.startswith('Image size (X x Y x C) : '):
            image_size = [str.split(" ")[8], str.split(" ")[10], str.split(" ")[12]]
        if str.startswith('Database : '):
            database = str.split(" : ")[1].replace("\"","")
        if str.startswith('Objects with ground truth : '):
            objects_with_ground_truth = str.split(" : ")[1].split(" ")[0]

    # remove header parameters from list
    content = content[4:]

    # divide list in sublists corresponding to the number of persons
    person_list =  [content[i:i+3] for i in range(0, len(content), 3)]

    # getting a list with bounding boxes (consider improve syntax)
    person_list_bounding_boxes = [person_list[i][2].split(":")[1].replace(" ","").replace("(","").replace(")","").replace("-",",").split(",") for i in range(0,len(person_list))]


    return [image_size, objects_with_ground_truth, person_list_bounding_boxes]


