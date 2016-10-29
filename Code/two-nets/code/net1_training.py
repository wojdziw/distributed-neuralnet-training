import caffe
import os
import numpy as np

solver = caffe.get_solver('../models/net1_solver.prototxt')

maxIter = 3
stepPerIter = 1

net2_iteration = -1

for net2_iteration in range(maxIter):
