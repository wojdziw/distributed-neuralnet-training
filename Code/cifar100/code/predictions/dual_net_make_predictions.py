# Adapted from Adil Moujahid
# https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial

import glob
import cv2
import caffe
import numpy as np

GPU_ID = 1
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

# Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../../input/test/*jpg")]
test_img_labels = np.load("../../input/test/labels.npy")

# Constants
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
ITERATION_NUMBER = 2000
NO_SAMPLES = len(test_img_paths)

def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

# Read model architecture and trained model's weights
net1 = caffe.Net('../../models/net1_deploy_def.prototxt',
                '../../snapshots/net1_iter_'+str(ITERATION_NUMBER)+'.caffemodel',
                caffe.TEST)

net2 = caffe.Net('../../models/net2_deploy_def.prototxt',
                '../../snapshots/net2_iter_'+str(ITERATION_NUMBER)+'.caffemodel',
                caffe.TEST)

# Define image transformers
transformer1 = caffe.io.Transformer({'data': net1.blobs['data'].data.shape})
transformer1.set_transpose('data', (2,0,1))

transformer2 = caffe.io.Transformer({'data2': net2.blobs['data2'].data.shape})
transformer2.set_transpose('data2', (2,0,1))



# Running the predictions
noCorrect = 0.0
for i in range(NO_SAMPLES):
    # Monitoring the progress
    if i%500 == 0:
        print str(i) + "/" + str(NO_SAMPLES) + " completed"

    # Loading the image
    img_path = test_img_paths[i]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, IMAGE_WIDTH, IMAGE_HEIGHT)

    # Preprocessing the image to feed it into the model
    net1.blobs['data'].data[...] = transformer1.preprocess('data', img)

    # Running the forward prop on net1 and retrieving the output of pool2
    out = net1.forward()
    data_pool2 = net1.blobs['pool2'].data[0]

    # Preparing the data to be fed into net2
    processedImage =  np.zeros([8,8,256])
    for j in range(256):
        processedImage[:,:,j] = data_pool2[j,:,:]

    # Preprocessing the data fed into net2
    net2.blobs['data2'].data[...] = transformer2.preprocess('data2', processedImage)

    # Running the forward prop on net2 and retrieving the class probabilities
    out = net2.forward()
    pred_probas = net2.forward()['prob']

    # Checking for correctness
    img_number = img_path.split("img")[1].split(".")[0]
    if pred_probas.argmax() == test_img_labels[int(img_number)][0]:
        noCorrect += 1

# Calculating the accuracy
accuracy = noCorrect/NO_SAMPLES
print accuracy
