import caffe
import os
import numpy as np
import time

GPU_ID = 1
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.get_solver('../models/net1_solver.prototxt')

maxIter = 2000
epochIter = 200
noEpochs = 10
stepPerIter = 1
learningRate = 0.00000001

losses = np.zeros(epochIter*noEpochs)

net2_epoch = -1

for net1_epoch in range(noEpochs):

	while(net1_epoch != net2_epoch):
		time.sleep(50)
		print "waiting..."
		try:
			net2_iteration = int(np.load("../comms/net2_iteration.npy"))
		except:
			pass

	while True:
		try:
			net2_data = np.load("../comms/data_conv3p.npy")
		except:
			pass
		else:
			break

	for iteration in range(epochIter)
		# copying the output of net2 to net1
		for i in range(data_conv3p.shape[0]):
			solver.net.blobs['conv3p'].data[i] = net2_data[i]

		# solver.step(1)

		# backprop and weight update
		solver.net.forward()
		
		solver.net.backward()
		for layer in solver.net.layers:
	    		for blob in layer.blobs:
	        		blob.data[...] -= learningRate*blob.diff

		if iteration%100==0:
			solver.net.save('../models/snapshots/net1_iter_'+str(net1_epoch*epochIter+net1_iteration)+'.caffemodel')

		losses[net1_epoch*epochIter+iteration] = float(solver.net.blobs['loss'].data)
		np.save('../models/snapshots/net1_losses', losses)



