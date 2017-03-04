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
ITERATION_NUMBER = 1500
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
net = caffe.Net('../../models/net12_deploy_def.prototxt',
                '../../snapshots/net12_iter_'+str(ITERATION_NUMBER)+'.caffemodel',
                caffe.TEST)

# Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))



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
    net.blobs['data'].data[...] = transformer.preprocess('data', img)

    # Running the forward prop and retrieving the class probabilities
    out = net.forward()
    pred_probas = out['prob']

    # Checking for correctness
    img_number = img_path.split("img")[1].split(".")[0]
    if pred_probas.argmax() == test_img_labels[int(img_number)][0]:
        noCorrect += 1

# Calculating the accuracy
accuracy = noCorrect/NO_SAMPLES
print accuracy
