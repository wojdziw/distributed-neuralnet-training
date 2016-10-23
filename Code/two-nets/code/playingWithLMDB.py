import os
import glob
import random
import numpy as np
import cv2

from genericCreateLMDB import *


train_data = [img for img in glob.glob("../input/train/*jpg")]
random.shuffle(train_data)

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

noImages = len(train_data)

images = np.zeros([noImages, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
labels = np.zeros(noImages)

for i, dir in enumerate(train_data):
	image = cv2.imread(train_data[i], cv2.IMREAD_COLOR)
	image = transform_img(image, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	print image.shape
	images[i] = image
	print images.shape

	if 'cat' in train_data[i]:
            labels[i] = 0
        else:
            labels[i] = 1

createLMDB(images, labels, '../input/train_lmdb')
