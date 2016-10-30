import caffe
import os
import numpy as np
import time
from subprocess import call

call(["rm", "../comms/*"])
solver = caffe.get_solver('../models/net2_solver.prototxt')
maxIter = 2
stepPerIter = 1

net1_iteration = -1

for net2_iteration in range(maxIter):
	
	########## START OF AN ITERATION ##########

	########## NET 1 FORWARD PASS START ##########

	# We can only start an iteration once we have data ready from net1
	print "Starting iteration " + str(net2_iteration) + " and waiting for net1's parameters"
	while(net1_iteration != net2_iteration):
		# tiny delay to prevent accessing an open file
		time.sleep(1)	
		if os.path.exists("../comms/net1_iteration.npy"):
			net1_iteration = int(np.load("../comms/net1_iteration.npy"))

	# Loading the parameters from net1
	data_pool2 = np.load("../comms/data_pool2.npy")

	# copying the net1 parameters into data of net 2
	for i in range(data_pool2.shape[0]):
		solver.net.blobs['data2'].data[i] = data_pool2[i]

	########## NET 1 FORWARD PASS FINISH ##########

	########## NET 2 FORWARD AND BACKWARD PASS START ##########
	
	solver.step(1)	

	# data computed by each of the layers
	data_input = solver.net.blobs['data2'].data
	data_conv3p = solver.net.blobs['conv3p'].data
	data_conv4 = solver.net.blobs['conv4'].data

	# parameters computed by each of the layers
	params_conv3p = solver.net.params['conv3p'][0].data
	params_conv4 = solver.net.params['conv4'][0].data
	
	# save the data produced by conv2 and give it to the other net
	np.save("../comms/data_conv3p", data_conv3p)

	########## NET 2 FORWARD AND BACKWARD PASS FINISH ##########
	
	########## NET 1 BACKWARD PASS START  ##########

	np.save('../comms/net2_iteration', net2_iteration)

	########## NET 1 BACKWARD PASS FINISH ##########

