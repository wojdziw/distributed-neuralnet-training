import caffe
import os
import numpy as np
import time

solver = caffe.get_solver('../models/net1_solver.prototxt')

maxIter = 2 
stepPerIter = 1

net2_iteration = -1

for net1_iteration in range(maxIter):
	
	########## START OF AN ITERATION ##########	

	########## NET 1 FORWARD PASS START ##########
	
	solver.net.forward()

	image_data = solver.net.blobs['data'].data
	data_conv1 = solver.net.blobs['conv1'].data
	data_pool1 = solver.net.blobs['pool1'].data
	data_conv2 = solver.net.blobs['conv2'].data
	data_pool2 = solver.net.blobs['pool2'].data
	data_conv3 = solver.net.blobs['conv3'].data
		
	# save the output of the second convolutional layer (+pool) into a file
	np.save('../comms/data_pool2', data_pool2)
	
	# send net2 info to start its computation
	np.save('../comms/net1_iteration', net1_iteration)

	########## NET 1 FORWARD FINISH ##########
		
	######### NET 2 FORWARD AND BACKWARD PASS START ##########
	
	# check if net2 has finished its computation
	while(net1_iteration != net2_iteration):
		time.sleep(1)	
		if os.path.exists("../comms/net2_iteration.npy"):
			net2_iteration = int(np.load("../comms/net1_iteration.npy"))
		
	########## NET 2 FORWARD AND BACKWARD PASS FINISH ##########

	########## NET 2 BACKWARD PASS START ##########
 	
	# load the data produced by the second net
	data_conv_3p = np.load("../comms/data_conv3p.npy")
	
	# not yet until we don't know how to define loss function appropriately
	# solver.net.backward()

	########## NET 2 BACKWARD PASS FINISH ##########

	########## END OF THE ITERATION ##########
	
	

