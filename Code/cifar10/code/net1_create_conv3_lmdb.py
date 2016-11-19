'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

def transform_img(img, img_width, img_height):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label, no_filters):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=no_filters,
        width=image_width,
        height=image_height,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def create_lmdb(input_path, output_path, image_width, image_height, no_filters):

    os.system('rm -rf  ' + output_path)

    data = [img for img in glob.glob(input_path)]
    np.save('../comms/image_data', data)

    #Shuffle data
    random.shuffle(data)

    in_db = lmdb.open(output_path, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, img_path in enumerate(data):
            if in_idx %  6 == 0:
                continue
            img = np.zeros([image_height, image_width, no_filters])
	    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	    #img = transform_img(img, image_width, image_height)
            
            datum = make_datum(img, 1, no_filters)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
            print '{:0>5d}'.format(in_idx) + ':' + img_path
    in_db.close()

    print '\nFinished processing all images'

    return 0

#Size of images
image_width = 13
image_height = 13
no_filters = 384
create_lmdb("../input/train/*jpg","../input/net1_lmdb_conv3", image_width, image_height, no_filters)
