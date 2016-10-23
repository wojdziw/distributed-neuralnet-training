import os
import glob
import random
import numpy as np
import cv2

from genericCreateLMDB import *


train_data = [img for img in glob.glob("../input/train/*jpg")]
random.shuffle(train_data)

noImages = len(train_data)

image0 = cv2.imread(train_data[0], cv2.IMREAD_COLOR)
imageShape = image0.shape

images = np.zeros([noImages, imageShape[0],imageShape[1], imageShape[2]])
labels = np.zeros(noImages)

for i, dir in enumerate(train_data):
	image = cv2.imread(train_data[i], cv2.IMREAD_COLOR)
	print image.shape
	images[i] = image
	print images.shape

	if 'cat' in train_data[i]:
            labels[i] = 0
        else:
            labels[i] = 1

createLMDB(images, labels, '../input/train_lmdb')
