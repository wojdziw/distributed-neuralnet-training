import caffe
import os
import numpy as np
import time

GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.get_solver('../../models/net12_solver.prototxt')
# this is a comment
# another one
maxIter = 1000
stepPerIter = 1

losses = np.zeros(maxIter)

for net1_iteration in range(maxIter):
	print "ITERATION " + str(net1_iteration)
	solver.step(1)
	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
        np.save('../../models/snapshots/net12_losses', losses)
