import caffe
import os
import numpy as np
import time

solver = caffe.get_solver('../models/net1_solver.prototxt')

maxIter = 1000 
stepPerIter = 1
learningRate = 0.00000001

losses = np.zeros(maxIter)
ithoughtlosses = np.zeros(maxIter)

net2_iteration = -1

for net1_iteration in range(maxIter):
	
	print "###############################"
	print "Iteration " + str(net1_iteration) + ": starting..."
	print "Iteration " + str(net1_iteration) + ": net1 forward pass start"
	
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



	print "Iteration " + str(net1_iteration) + ": net1 forward pass finish"
		
	print "Iteration " + str(net1_iteration) + ": net2 forward and back pass start"
	
	# check if net2 has finished its computation
	while(net1_iteration != net2_iteration):
		time.sleep(10)
		print "waiting..."	
		if os.path.exists("../comms/net2_iteration.npy"):
			net2_iteration = int(np.load("../comms/net2_iteration.npy"))
		
	print "Iteration " + str(net1_iteration) + ": net2 forward and back pass finish"

	print "Iteration " + str(net1_iteration) + ": net1 back pass start"

 	
	# load the data produced by the second net
	net2_data = np.load("../comms/data_conv3p.npy")
	
	# copying the output of net2 to net1
	for i in range(data_conv3p.shape[0]):
		solver.net.blobs['conv3p'].data[i] = net2_data[i]
	
	# not sure how to only do backprop so that everything is updated properly but hope this works
	solver.net.backward()
	for layer in solver.net.layers:
    		for blob in layer.blobs:
        		blob.data[...] -= learningRate*blob.diff
	
	print "Sum of conv3p is: " + str(np.sum(solver.net.blobs['conv3p'].data))
	
	difference = solver.net.blobs['conv3p'].data-solver.net.blobs['conv3'].data
	loss = np.sum(difference**2)/solver.net.blobs['conv3p'].num/2
	
	# save the value of the loss
	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
	ithoughtlosses[net1_iteration] = loss 

	print "Loss is: " + str(float(solver.net.blobs['loss'].data))
	print "And I thought it would be:" + str(loss)
	print "Iteration " + str(net1_iteration) + ": net1 back pass finish"
	
	np.save('../models/snapshots/net1_losses', losses)
	np.save('../models/snapshots/net1_perceived_losses', ithoughtlosses)
	if net1_iteration%100==0: #and net1_iteration>0:
		solver.net.save('../models/snapshots/net1_iter'+str(net1_iteration)+'.caffemodel')	

