import caffe
import os
import numpy as np
import time

solver = caffe.get_solver('../models/net12_solver.prototxt')

maxIter = 1000
stepPerIter = 1

losses = np.zeros(maxIter)

for net1_iteration in range(maxIter):
	print "ITERATION " + str(net1_iteration)
	solver.step(1)
	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
        np.save('../models/snapshots/net12_losses', losses)
	
