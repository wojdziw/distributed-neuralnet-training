# Adapted from Adil Moujahid
# https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial

import os
import glob
import random
import numpy as np
import cv2
import caffe
import lmdb
from caffe.proto import caffe_pb2

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
NO_FILTERS = 3
TRAIN_IMAGES_PATH = "../../input/train/*jpg"
TRAIN_OUTPUT_PATH = "../../input/net1_train_lmdb"
TRAIN_LABELS_PATH = "../../input/train/labels.npy"
TEST_IMAGES_PATH = "../../input/test/*jpg"
TEST_OUTPUT_PATH = "../../input/net1_test_lmdb"
TEST_LABELS_PATH = "../../input/test/labels.npy"

def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

def make_datum(img, label, img_width, img_height, no_filters):
    # Make sure the channels are flipped to conform to the cv2 BGR format
    return caffe_pb2.Datum(
        channels=no_filters,
        width=img_width,
        height=img_height,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def create_lmdb(labels_path, input_path, output_path, img_width, img_height, no_filters):
    # Loading the input
    labels = np.load(labels_path)
    data = [img for img in glob.glob(input_path)]

    # Cleaning up the output
    os.system('rm -rf  ' + output_path)

    # Shuffle data
    random.shuffle(data)

    # Produce the LMDB
    in_db = lmdb.open(output_path, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(data):
    	    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    	    img = transform_img(img, IMAGE_WIDTH, IMAGE_HEIGHT)
    	    img_number = img_path.split("img")[1].split(".")[0]
    	    label = labels[int(img_number)][0]
            datum = make_datum(img, label, img_width, img_height, no_filters)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())

            if in_idx%500==0:
    			print str(in_idx) + "/" + str(len(data)) + " completed"
    in_db.close()

    print "Finished processing all images"
    return 0

# Producing the train and test LMDBs.
create_lmdb(TRAIN_LABELS_PATH,TRAIN_IMAGES_PATH,TRAIN_OUTPUT_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, NO_FILTERS)
create_lmdb(TEST_LABELS_PATH,TEST_IMAGES_PATH,TEST_OUTPUT_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, NO_FILTERS)
