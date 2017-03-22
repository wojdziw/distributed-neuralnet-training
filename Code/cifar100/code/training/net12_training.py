import caffe
import os
import numpy as np
import time

GPU_ID = 0
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.get_solver('../../models/net12_solver.prototxt')
maxIter = 3000
stepPerIter = 1

losses = np.zeros(maxIter+1)

start_time = time.time()

for net1_iteration in range(maxIter+1):
	print "ITERATION " + str(net1_iteration)
	solver.step(1)
	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
        np.save('../../snapshots/net12_losses', losses)

end_time = time.time()
total_non_idle_time = end_time-start_time
print "Total non-idle time is " + str(total_non_idle_time)
np.save('../../snapshots/net12_time_taken', total_non_idle_time)
