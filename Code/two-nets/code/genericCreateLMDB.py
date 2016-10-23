'''
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid, adapted by Wojciech Dziwulski
'''


import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb


def createLMDB(images, labels, outputDirectory):
    
    os.system('rm -rf  ' + outputDirectory)

    print 'Creating train_lmdb'

    in_db = lmdb.open(outputDirectory, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
    	for in_idx, img in enumerate(images):
    	    if in_idx %  6 == 0:
                continue
    	    
            label = int(labels[in_idx])
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
    	    print in_idx
    	print "Finished processing images, now other stuff..."

    print "Nearly finished"
    in_db.close()

    print '\nFinished processing all images'

    return 0

def transform_img(img, img_width, img_height):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=img.shape[1],
        height=img.shape[0],
        label=label,
        data=np.rollaxis(img, 2).tostring())
