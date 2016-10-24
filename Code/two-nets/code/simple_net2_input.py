import numpy as np
import cv2

generate_white_input(no_filters, image_size, file_names, output_path):
	image = np.zeros([no_filters, image_size[0], image_size[1]])

	for file_name in file_names:
		cv2.imwrite(file_name, image)

	return 0

no_filters = 96
image_size = [13,13]
file_names = np.load('../comms/image_data.npy')
output_path = "../input/net2_train"

generate_white_input(no_filters, image_size, file_names, output_path)