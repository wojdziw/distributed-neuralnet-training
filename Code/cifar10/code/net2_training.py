import caffe
import os
import numpy as np
import time

GPU_ID = 2
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.SGDSolver('../models/net2_solver.prototxt')

maxIter = 2000
epochIter = 100
noEpochs = 10

losses = np.zeros(maxIter)

net1_iteration = -1
np.save('../comms/net1_iteration', net1_iteration)
np.save('../comms/net2_iteration', -1)

for net2_iteration in range(maxIter):

	if ((max(0,net2_iteration-10)/epochIter)%2==0 and net2_iteration>=10):
		print "Iteration " + str(net2_iteration) + ": idling"

		while(net1_iteration != net2_iteration):
			# tiny delay to prevent accessing an open file
			time.sleep(5)
			print "waiting..."
			try:
				net1_iteration = int(np.load("../comms/net1_iteration.npy"))
			except:
				pass

		data_conv3p = solver.net.blobs['conv3p'].data
		np.save("../comms/data_conv3p", data_conv3p)
		np.save('../comms/net2_iteration', net2_iteration)
		losses[net2_iteration] = float(solver.net.blobs['loss'].data)
		np.save('../models/snapshots/net2_losses', losses)
		continue

	print "###############################"
	print "Iteration " + str(net2_iteration) + ": starting..."
	print "Iteration " + str(net2_iteration) + ": net1 forward pass start"

	# We can only start an iteration once we have data ready from net1
	while(net1_iteration != net2_iteration):
		# tiny delay to prevent accessing an open file
		time.sleep(5)
		print "waiting..."
		try:
			net1_iteration = int(np.load("../comms/net1_iteration.npy"))
		except:
			pass

	# loading the parameters from net1
	while True:
		try:
			data_pool2 = np.load("../comms/data_pool2.npy")
			labels = np.load("../comms/net1_labels.npy")
		except:
			pass
		else:
			break

	net = solver.net
	net.set_input_arrays(data_pool2,labels)

	print "Iteration " + str(net2_iteration) + ": net1 forward pass finish"

	print "Iteration " + str(net2_iteration) + ": net2 forward and back pass start"

	# run forward and back prop
	solver.step(1)

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

	losses[net2_iteration] = float(solver.net.blobs['loss'].data)
	np.save('../models/snapshots/net2_losses', losses)
