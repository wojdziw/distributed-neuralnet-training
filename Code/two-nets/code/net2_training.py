import caffe
import os
import numpy as np
import time

solver = caffe.get_solver('../models/net2_solver.prototxt')
maxIter = 1
stepPerIter = 1
learningRate = 0.001

losses = np.zeros(maxIter)

net1_iteration = -1
np.save('../comms/net1_iteration', net1_iteration)
np.save('../comms/net2_iteration', -1)

for net2_iteration in range(maxIter):
	
	print "########## START OF THE ITERATION " + str(net2_iteration) +  " ##########"

	print "########## NET 1 FORWARD PASS START ##########"

	# We can only start an iteration once we have data ready from net1
	print "Starting iteration " + str(net2_iteration) + " and waiting for net1's parameters"
	while(net1_iteration != net2_iteration):
		# tiny delay to prevent accessing an open file
		time.sleep(10)	
		print "waiting..."
		if os.path.exists("../comms/net1_iteration.npy"):
			net1_iteration = int(np.load("../comms/net1_iteration.npy"))

	# Loading the parameters from net1
	data_pool2 = np.load("../comms/data_pool2.npy")
	labels = np.load("../comms/net1_labels.npy")	

	# copying the net1 output into data of net 2
	for i in range(data_pool2.shape[0]):
		solver.net.blobs['data2'].data[i] = data_pool2[i]

	# copying the labels
	for i in range(len(labels)):
		solver.net.blobs['label'].data[i] = labels[i]

	print solver.net.blobs['data2'].data[0][0][0]

	print "########## NET 1 FORWARD PASS FINISH ##########"

	print "########## NET 2 FORWARD AND BACKWARD PASS START ##########"
	
	# this overwrites my data layer change	
	solver.net.forward()	
	
	print solver.net.blobs['data2'].data[0][0][0]
	
	for i in range(data_pool2.shape[0]):
                solver.net.blobs['data2'].data[i] = data_pool2[i]
	
	solver.net.backward()

	print solver.net.blobs['data2'].data[0][0][0]

	for layer in solver.net.layers:
                for blob in layer.blobs:
                        blob.data[...] -= learningRate*blob.diff

	# data computed by each of the layers
	data_input = solver.net.blobs['data2'].data
	data_conv3p = solver.net.blobs['conv3p'].data
	data_conv4 = solver.net.blobs['conv4'].data
	losses[net2_iteration] = float(solver.net.blobs['loss'].data)
	print "Loss is: " + str(float(solver.net.blobs['loss'].data))	

	# parameters computed by each of the layers
	params_conv3p = solver.net.params['conv3p'][0].data
	params_conv4 = solver.net.params['conv4'][0].data
	
	# save the data produced by conv2 and give it to the other net
	np.save("../comms/data_conv3p", data_conv3p)

	print "########## NET 2 FORWARD AND BACKWARD PASS FINISH ##########"
	
	print "########## NET 1 BACKWARD PASS START  ##########"

	np.save('../comms/net2_iteration', net2_iteration)

	print "########## NET 1 BACKWARD PASS FINISH ##########"

np.save('../models/snapshots/net2_losses', losses)
