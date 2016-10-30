import caffe
import os
import numpy as np
import time

solver = caffe.get_solver('../models/net1_solver.prototxt')

maxIter = 50 
stepPerIter = 1

net2_iteration = -1

for net1_iteration in range(maxIter):
	
	print "########## START OF THE ITERATION " + str(net1_iteration) + " ##########"	

	print "########## NET 1 FORWARD PASS START ##########"
	
	solver.net.forward()

	image_data = solver.net.blobs['data'].data
	labels = solver.net.blobs['label'].data
	data_conv1 = solver.net.blobs['conv1'].data
	data_pool1 = solver.net.blobs['pool1'].data
	data_conv2 = solver.net.blobs['conv2'].data
	data_pool2 = solver.net.blobs['pool2'].data
	data_conv3 = solver.net.blobs['conv3'].data
	data_conv3p = solver.net.blobs['conv3p'].data
		
	# save the output of the second convolutional layer (+pool) into a file
	np.save('../comms/data_pool2', data_pool2)
	
	# send net1 to net2 info to start its computation
	np.save('../comms/net1_iteration', net1_iteration)

	# send net1 labels to net2 to align the information
	np.save('../comms/net1_labels', labels)

	print "########## NET 1 FORWARD FINISH ##########"
		
	print "######### NET 2 FORWARD AND BACKWARD PASS START ##########"
	
	# check if net2 has finished its computation
	while(net1_iteration != net2_iteration):
		time.sleep(3)
		print "waiting..."	
		if os.path.exists("../comms/net2_iteration.npy"):
			net2_iteration = int(np.load("../comms/net2_iteration.npy"))
		
	print "########## NET 2 FORWARD AND BACKWARD PASS FINISH ##########"

	print "########## NET 2 BACKWARD PASS START ##########"
 	
	# load the data produced by the second net
	net2_data = np.load("../comms/data_conv3p.npy")
	
	# copying the output of net2 to net1
	for i in range(data_conv3p.shape[0]):
		solver.net.blobs['conv3p'].data[i] = net2_data[i]
	
	solver.net.backward()
	
	print "########## NET 2 BACKWARD PASS FINISH ##########"

	print "########## END OF THE ITERATION ##########"
	
	

