#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2015, Joao Quintas
# All rights reserved.
#
# Adapted from Philipp Wagner <bytefish[at]gmx[dot]de>.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the author nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys, os

sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, SpatialHistogram, Identity, PCA
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP

# import for measure execution time of small code snippets (https://docs.python.org/3.4/library/timeit.html)
import timeit


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
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as error:
                    print ("I/O error({0}): {1}".format(error.errno, error.strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    return [X, y]


if __name__ == "__main__":
    """
        General structure of the algorithm follows:
        1. read images
            1.1. resize and reformat images if needed
        2. obtain mean image
        3. apply pca
        4. obtain k largest for eigenvectors
        5. obtain k largest eigenfaces
        6. project new face in the eigenfaces space
    """

    # This is where we write the images, if an output_dir is given
    # in command line:
    out_dir = None

    # You'll need at least a path to your image data, please see
    # the tutorial coming with this source code on how to prepare
    # your image data:
    if len(sys.argv) < 2:
        print ("USAGE: eigenfaces_nocontext.py </path/to/images>")
        sys.exit()
    # Now read in the image data. This must be a valid path!
    [X, y] = read_images(sys.argv[1])


    # Then set up a handler for logging:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # Add handler to facerec modules, so we see what's going on inside:
    logger = logging.getLogger("facerec")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)





    # Define the PCA as Feature Extraction method:
    feature = PCA()

    # Define a 1-NN classifier with Euclidean Distance:
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)

    # Define the model as the combination
    my_model = PredictableModel(feature=feature, classifier=classifier)

    # Compute the Eigenfaces on the given data (in X) and labels (in y):
    t_train_model = timeit.timeit(stmt='my_model.compute(X, y)', setup='from __main__ import my_model, X, y', number=1)
    print("\ncomputing model for {} images from {} individuals: {}s\n".format(len(X), len(set(y)) ,t_train_model))

    # We then save the model, which uses Pythons pickle module:
    save_model('model.pkl', my_model)
    model = load_model('model.pkl')





    # Then turn the first (at most) 16 eigenvectors into grayscale
    # images (note: eigenvectors are stored by column!)
    E = []
    for i in range(min(model.feature.eigenvectors.shape[1], 16)):
        e = model.feature.eigenvectors[:, i].reshape(X[0].shape)
        E.append(minmax_normalize(e, 0, 255, dtype=np.uint8))

    # Plot them and store the plot to "eigenfaces.png"
    subplot(title="Eigenfaces", images=E, rows=4, cols=4, sptitle="Eigenfaces", colormap=cm.gray,
            filename="eigenfaces.png")





    # Perform a 10-fold cross validation
    k=10
    cv = KFoldCrossValidation(model, k)
    t_validation = timeit.timeit(stmt='cv.validate(X, y)', setup='from __main__ import cv, X, y', number=1)
    print("\nk-fold cross validation (k={}): {}s\n".format(k, t_validation))

    # And print the result:
    print("\n\nResults:\n")
    cv.print_results()