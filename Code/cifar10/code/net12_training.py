import caffe
import os
import numpy as np
import time

solver = caffe.get_solver('../models/net12_solver.prototxt')

maxIter = 10
stepPerIter = 1

net2_iteration = -1

for net1_iteration in range(maxIter):
	print "ITERATION " + str(net1_iteration)
	solver.step(1)
	
