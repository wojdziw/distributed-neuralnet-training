import caffe
import os
import numpy as np
import time

os.system("mkdir ../../comms")
os.system("mkdir ../../snapshots")

GPU_ID = 2
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

solver = caffe.get_solver('../../models/net2_solver.prototxt')

net1_seq_size = 50
net2_seq_size = 150
no_seqs = 20

losses = np.zeros(no_seqs*(net1_seq_size+net2_seq_size)+1)
total_non_idle_time = 0

net1_iteration = -1
np.save('../../comms/net1_iteration', -1)
np.save('../../comms/net2_iteration', -1)
np.save("../../comms/data_conv3p", solver.net.blobs['conv3p'].data)

for net2_iteration in range(no_seqs*(net1_seq_size+net2_seq_size)+1):

	if (net2_iteration%(net1_seq_size+net2_seq_size)<net1_seq_size and net2_iteration>=10):
		print "Iteration " + str(net2_iteration) + ": idling"

		while(net1_iteration != net2_iteration):
			# tiny delay to prevent accessing an open file
			time.sleep(5)
			print "waiting..."
			try:
				net1_iteration = int(np.load("../../comms/net1_iteration.npy"))
			except:
				pass

	else:
		start_time = time.time()

		# We can only start an iteration once we have data ready from net1
		while(net1_iteration != net2_iteration):
			# tiny delay to prevent accessing an open file
			time.sleep(5)
			print "waiting..."
			try:
				net1_iteration = int(np.load("../../comms/net1_iteration.npy"))
			except:
				pass

		# loading the parameters from net1
		while True:
			try:
				data_pool2 = np.load("../../comms/data_pool2.npy")
				labels = np.load("../../comms/net1_labels.npy")
			except:
				pass
			else:
				break

		net = solver.net
		net.set_input_arrays(data_pool2,labels)

		# run forward and back prop
		solver.step(1)

		end_time = time.time()
		total_non_idle_time += end_time-start_time

	np.save("../../comms/data_conv3p", solver.net.blobs['conv3p'].data)

	losses[net2_iteration] = float(solver.net.blobs['loss'].data)
	np.save('../../snapshots/net2_losses', losses)

	if net2_iteration%50==0:
		solver.net.save('../../snapshots/net2_iter_'+str(net1_iteration)+'.caffemodel')

	print "Iteration " + str(net2_iteration) + ". Loss is: " + str(float(solver.net.blobs['loss'].data))

	np.save('../../comms/net2_iteration', net2_iteration)

print "Total non-idle time is " + str(total_non_idle_time)
np.save('../../snapshots/net2_time_taken', total_non_idle_time)
