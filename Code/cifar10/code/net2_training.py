import caffe
import os
import numpy as np
import time

solver = caffe.SGDSolver('../models/net2_solver.prototxt')
maxIter = 1000
stepPerIter = 1
learningRate = 0.00000001

losses = np.zeros(maxIter)

net1_iteration = -1
np.save('../comms/net1_iteration', net1_iteration)
np.save('../comms/net2_iteration', -1)

for net2_iteration in range(maxIter):

	print "###############################"
	print "Iteration " + str(net2_iteration) + ": starting..."
	print "Iteration " + str(net2_iteration) + ": net1 forward pass start"

	# We can only start an iteration once we have data ready from net1
	while(net1_iteration != net2_iteration):
		# tiny delay to prevent accessing an open file
		time.sleep(10)
		print "waiting..."
		if os.path.exists("../comms/net1_iteration.npy"):
			net1_iteration = int(np.load("../comms/net1_iteration.npy"))

	# loading the parameters from net1
	data_pool2 = np.load("../comms/data_pool2.npy")
	labels = np.load("../comms/net1_labels.npy")

	# copying the new parameters into the data layers
	# copying the net1 output into data of net 2
	#for i in range(data_pool2.shape[0]):
		#solver.net.blobs['data2'].data[i] = data_pool2[i]

	# copying the labels
	#for i in range(len(labels)):
		#solver.net.blobs['label'].data[i] = labels[i]

	net = solver.net
	net.set_input_arrays(data_pool2,labels)

	print "Iteration " + str(net2_iteration) + ": net1 forward pass finish"

	print "Iteration " + str(net2_iteration) + ": net2 forward and back pass start"

	# run forward and back prop
	solver.step(1)
	'''
	solver.net.forward()
        solver.net.backward()
        for layer in solver.net.layers:
                for blob in layer.blobs:
                        blob.data[...] -= learningRate*blob.diff
	'''

	# data computed by each of the layers
	data_input = solver.net.blobs['data2'].data
	data_conv3p = solver.net.blobs['conv3p'].data
	data_conv4 = solver.net.blobs['conv4'].data
	losses[net2_iteration] = float(solver.net.blobs['loss'].data)
	print "Loss is: " + str(float(solver.net.blobs['loss'].data))

	# parameters computed by each of the layers
	params_conv3p = solver.net.params['conv3p'][0].data
	params_conv4 = solver.net.params['conv4'][0].data

	# save the data produced by conv3p and give it to the other net
	np.save("../comms/data_conv3p", data_conv3p)

	print "Iteration " + str(net2_iteration) + ": net2 forward and back pass finish"

	print "Iteration " + str(net2_iteration) + ": net1 back pass start"

	np.save('../comms/net2_iteration', net2_iteration)

	print "Iteration " + str(net2_iteration) + ": net1 back pass finish"

	np.save('../models/snapshots/net2_losses', losses)
