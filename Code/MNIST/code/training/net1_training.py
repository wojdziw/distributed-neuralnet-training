import caffe
import os
import numpy as np
import time

GPU_ID = 1
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.get_solver('../../models/net1_solver.prototxt')

net1_seq_size = 50
net2_seq_size = 150

no_seqs = 20

losses = np.zeros(no_seqs*(net1_seq_size+net2_seq_size)+1)
total_non_idle_time = 0
net2_iteration = -1

for net1_iteration in range(no_seqs*(net1_seq_size+net2_seq_size)+1):

	if (net1_iteration%(net1_seq_size+net2_seq_size)>=net1_seq_size):

		print "Iteration " + str(net1_iteration) + ": idling"

		solver.net.forward()
		np.save('../../comms/data_pool2', solver.net.blobs['pool2'].data)
		np.save('../../comms/net1_labels', solver.net.blobs['label'].data)
		np.save('../../comms/net1_iteration', net1_iteration)

	else:
		start_time = time.time()

		data_conv3p = solver.net.blobs['conv3p'].data
		# load the data produced by the second net
		while True:
			try:
				net2_data = np.load("../../comms/data_conv3p.npy")
			except:
				pass
			else:
				break

		# copying the output of net2 to net1
		for i in range(data_conv3p.shape[0]):
			solver.net.blobs['conv3p'].data[i] = net2_data[i]

		# backprop and weight update
		solver.step(1)
		np.save('../../comms/net1_iteration', net1_iteration)

		end_time = time.time()
		total_non_idle_time += end_time-start_time

	# check if net2 has finished its computation
	while(net1_iteration != net2_iteration):
		time.sleep(5)
		print "waiting..."
		try:
			net2_iteration = int(np.load("../../comms/net2_iteration.npy"))
		except:
			pass

	# save the value of the loss
	losses[net1_iteration] = float(solver.net.blobs['loss'].data)
	np.save('../../snapshots/net1_losses', losses)

	if net1_iteration%50==0: #and net1_iteration>0:
		solver.net.save('../../snapshots/net1_iter_'+str(net1_iteration)+'.caffemodel')

	print "Iteration " + str(net1_iteration) + ". Loss is: " + str(float(solver.net.blobs['loss'].data))

print "Total non-idle time is " + str(total_non_idle_time)
np.save('../../snapshots/net1_time_taken', total_non_idle_time)
