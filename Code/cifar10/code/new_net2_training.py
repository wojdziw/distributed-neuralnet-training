import caffe
import os
import numpy as np
import time

GPU_ID = 2
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.SGDSolver('../models/net2_solver.prototxt')

epochIter = 200
noEpochs = 10

losses = np.zeros(maxIter)

net1_iteration = -1
np.save('../comms/net1_iteration', net1_iteration)
np.save('../comms/net2_iteration', -1)

for net2_iteration in range(noEpochs*epochIter):

    while(net1_iteration != net2_iteration):
		# tiny delay to prevent accessing an open file
		time.sleep(5)
		print "waiting..."
		try:
			net1_iteration = int(np.load("../comms/net1_iteration.npy"))
		except:
			pass

	while True:
		try:
			data_pool2 = np.load("../comms/data_pool2.npy")
			labels = np.load("../comms/net1_labels.npy")
		except:
			pass
		else:
			break

    if ((max(0,net1_iteration-10)/epochIter)%2==0 and net2_iteration>10):
        # net2 idle
        print "Iteration " + str(net1_iteration) + ": idling"
	else:
        print "Iteration " + str(net1_iteration) + ": full net2 step"
		# run full sweep on net2
        net = solver.net
    	net.set_input_arrays(data_pool2,labels)

		solver.step(1)

        data_conv3p = solver.net.blobs['conv3p'].data
        np.save("../comms/data_conv3p", data_conv3p)

	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
	np.save('../models/snapshots/net1_losses', losses)

    np.save('../comms/net2_iteration', net2_iteration)
