import caffe
import os
import numpy as np
import time

GPU_ID = 1
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.get_solver('../models/net1_solver.prototxt')

epochIter = 100
noEpochs = 20
learningRate = 0.00000000001

losses = np.zeros(epochIter*noEpochs)
net2_iteration = -1

for net1_iteration in range(noEpochs*epochIter):

	solver.net.forward()

	labels = solver.net.blobs['label'].data
	data_pool2 = solver.net.blobs['pool2'].data
	np.save('../comms/data_pool2', data_pool2)
	np.save('../comms/net1_iteration', net1_iteration)
	np.save('../comms/net1_labels', labels)

	if ((max(0,net1_iteration-10)/epochIter)%2==1):
		print "Iteration " + str(net1_iteration) + ": idling"

		while(net1_iteration != net2_iteration):
			time.sleep(5)
			print "waiting..."
			try:
				net2_iteration = int(np.load("../comms/net2_iteration.npy"))
			except:
				pass

	else:
		data_conv3p = solver.net.blobs['conv3p'].data

		# check if net2 has finished its computation
		while(net1_iteration != net2_iteration):
			time.sleep(5)
			print "waiting..."
			try:
				net2_iteration = int(np.load("../comms/net2_iteration.npy"))
			except:
				pass

		# load the data produced by the second net
		while True:
			try:
				net2_data = np.load("../comms/data_conv3p.npy")
			except:
				pass
			else:
				break

		# copying the output of net2 to net1
		for i in range(data_conv3p.shape[0]):
			solver.net.blobs['conv3p'].data[i] = net2_data[i]

		# backprop and weight update
		solver.net.backward()
		for layer in solver.net.layers:
	    		for blob in layer.blobs:
	        		blob.data[...] -= learningRate*blob.diff


	# save the value of the loss
	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
	np.save('../models/snapshots/net1_losses', losses)

	print "Iteration " + str(net1_iteration) + "; Loss is: " + str(float(solver.net.blobs['loss'].data))

	if net1_iteration%100==0: #and net1_iteration>0:
		solver.net.save('../models/snapshots/net1_iter_'+str(net1_iteration)+'.caffemodel')
