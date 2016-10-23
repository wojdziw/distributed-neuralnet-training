import caffe
import os
import numpy as np

solver = caffe.get_solver('../models/net2_solver.prototxt')

maxIter = 3
stepPerIter = 1

net1_iteration = -1

for net2_iteration in range(maxIter):
	print "Starting iteration " + str(net2_iteration + "and waiting for net1's parameters")
	while(net1_iteration != net2_iteration):
		if os.path.exists("net1_iteration.npy"):
			net1_iteration = int(np.load("net1_iteration.npy"))

	# np.load("parameters.npy")

	solver.step(1)

	# data computed by each of the layers
	originalData = solver.net.blobs['data2'].data
	dataConv4 = solver.net.blobs['conv4'].data
	dataPool4 = solver.net.blobs['pool4'].data
	dataConv5 = solver.net.blobs['conv5'].data
	dataPool5 = solver.net.blobs['pool5'].data
	dataConv6 = solver.net.blobs['conv6'].data
        dataPool6 = solver.net.blobs['pool6'].data
    
	# parameters computed by each of the layers
	paramsConv1 = solver.net.params['conv4'][0].data
	paramsConv2 = solver.net.params['conv5'][0].data
	paramsConv3 = solver.net.params['conv6'][0].data
	
	# save the data produced by conv2 and give it to the other net

	# the new data gets communicated and processed by the other net
	# what should be the target labels for this net beeeee	

	# after the other net is forward and back propped:
	# RUN ADMM

	np.save('net2_iteration', net2_iteration)
