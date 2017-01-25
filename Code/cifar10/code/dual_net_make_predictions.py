'''
Adapted from Adil Moujahid
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()

#Size of images
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

#Read model architecture and trained model's weights
net1 = caffe.Net('../models/net1_deploy_def.prototxt',
                '../models/snapshots/first-try/net1_iter200.caffemodel',
                caffe.TEST)

net2 = caffe.Net('../models/net2_deploy_def.prototxt',
                '../models/snapshots/first-try/net2_iter_200.caffemodel',
                caffe.TEST)

#Define image transformers
transformer1 = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
transformer1.set_transpose('data', (2,0,1))

transformer2 = caffe.io.Transformer({'data2': net2.blobs['data2'].data.shape})
transformer2.set_transpose('data2', (2,0,1))


#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../input/test/*jpg")]
test_img_labels = np.load("../input/test/labels.npy")

#Calculating the accuracy
noCorrect = 0.0

#Making predictions
test_ids = []
preds = []
noAnalysed = 1 #len(test_img_paths)

for i in range(noAnalysed):
    if i%5 == 0:
        print i
        print noCorrect/(i-1)

    img_path = test_img_paths[i]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net1.blobs['data'].data[...] = transformer1.preprocess('data', img)

    out = net1.forward()
    data_pool2 = net1.blobs['pool2'].data[0]

    # multiply input by one of the images and see if it gives something out
    print net1.blobs['data'].data.shape
    print "###################################################"
    print net1.params['conv1'][0].data.shape
    print "###################################################"
    print np.sum(net1.blobs['conv1'].data)

    processedImage =  np.zeros([8,8,256])

    for j in range(256):
        processedImage[:,:,j] = data_pool2[j,:,:]

    net2.blobs['data2'].data[...] = transformer2.preprocess('data2', processedImage)

    out = net2.forward()
    pred_probas = out['prob']

    preds = preds + [pred_probas.argmax()]
    img_number = img_path.split("img")[1].split(".")[0]

    #print img_path
    #print str(i) + " predicted label: " + str(pred_probas.argmax()) + ", true label: " + str(test_img_labels[int(img_number)][0])
    #print '-------'

    if pred_probas.argmax() == test_img_labels[int(img_number)][0]:
        noCorrect += 1

accuracy = noCorrect/noAnalysed
print accuracy
