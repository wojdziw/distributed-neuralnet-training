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
from subprocess import call

GPU_ID = 2
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

# The values here are very much depend on the architecture used
# They have to be found empirically by examining the size of the conv3 layer
# This is of course no ideal and can be considered a dirty hack
# Serious attempts to moving all this to a MemoryData layer were made
# With no success, due to the bugs in Caffe
IMAGE_WIDTH = 8
IMAGE_HEIGHT = 8
NO_FILTERS = 384
INPUT_PATH = "../../input/train/*jpg"
OUTPUT_TEST_PATH = "../../input/net1_conv3_test_lmdb"
OUTPUT_TRAIN_PATH = "../../input/net1_conv3_train_lmdb"

def make_datum(img, label, img_width, img_height, no_filters):
    # Make sure the channels are flipped to conform to the cv2 BGR format
    return caffe_pb2.Datum(
        channels=no_filters,
        width=img_width,
        height=img_height,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def create_lmdb(input_path, output_path, img_width, img_height, no_filters):
    # Loading the input - only for reference (we have to generate a good dummy)
    data = [img for img in glob.glob(input_path)]

    # Cleaning up the output
    os.system('rm -rf  ' + output_path)

    # Shuffle data
    random.shuffle(data)

    # Produce the LMDB
    in_db = lmdb.open(output_path, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(data):
            img = np.zeros([img_height, img_width, no_filters])
            datum = make_datum(img, 1, img_width, img_height, no_filters)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())

            if in_idx%500==0:
                print str(in_idx) + "/" + str(len(data)) + " completed"
    in_db.close()

    print "Finished processing all images"
    return 0

create_lmdb(INPUT_PATH, OUTPUT_TRAIN_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, NO_FILTERS)
print "Copying into the test dummy..."
call(["cp", "-r", OUTPUT_TRAIN_PATH, OUTPUT_TEST_PATH])
