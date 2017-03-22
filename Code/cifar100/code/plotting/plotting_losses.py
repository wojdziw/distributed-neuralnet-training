import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os.path

# User decision which plots to include
PLOT_DUAL = True
PLOT_SINGLE = True

net1_auto_yscale = True
net2_auto_yscale = False
net12_auto_yscale = True

net1_full_range = True
net2_full_range = True
net12_full_range = True

net1_without_steps = True
net2_without_steps = True

net1_seq_size = 50
net2_seq_size = 150
no_seqs = 40

# Checking if the loss OS paths exist
dual_path_exists = os.path.isfile("../../snapshots/net1_losses.npy")
single_path_exists = os.path.isfile("../../snapshots/net12_losses.npy")

def non_zero_entries(numbers):
	for i in range(numbers.size):
		if numbers[i]==0:
			break
	return i

def net2_losses_pruner(numbers, net1_seq_size, net2_seq_size, no_seqs):
	losses = np.zeros(no_seqs*net2_seq_size)
	counter = 0
	for net2_iteration in range(no_seqs*(net1_seq_size+net2_seq_size)+1):
		if (net2_iteration%(net1_seq_size+net2_seq_size)>=net1_seq_size):
			losses[counter] = numbers[net2_iteration]
			counter = counter+1
	return losses

def net1_losses_pruner(numbers, net1_seq_size, net2_seq_size, no_seqs):
	losses = np.zeros(no_seqs*net2_seq_size)
	counter = 0
	for net1_iteration in range(no_seqs*(net1_seq_size+net2_seq_size)+1):
		if (max(0,net1_iteration-10)%(net1_seq_size+net2_seq_size)<net1_seq_size and net1_iteration>=10):
			losses[counter] = numbers[net1_iteration]
			counter = counter+1
	return losses

# Plotting settings
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)

if PLOT_DUAL and dual_path_exists:

	y1 = np.load("../../snapshots/net1_losses.npy")
	y2 = np.load("../../snapshots/net2_losses.npy")

	if (net1_without_steps):
		y1 = net1_losses_pruner(y1, net1_seq_size, net2_seq_size, no_seqs)
	if (net2_without_steps):
		y2 = net2_losses_pruner(y2, net1_seq_size, net2_seq_size, no_seqs)

	# Calculating the range of the function to plot, can be overwritten
	n1 = non_zero_entries(y1)
	n2 = non_zero_entries(y2)
	if (net1_full_range and not net1_without_steps):
		n1 = no_seqs*(net1_seq_size+net2_seq_size)+1
	if (net2_full_range and not net2_without_steps):
		n2 = no_seqs*(net1_seq_size+net2_seq_size)+1
	x1 = range(n1)
	x2 = range(n2)
	y1 = y1[0:n1]
	y2 = y2[0:n2]

	plt.figure(1)

	# Plotting the output of net1
	plt.subplot(311)
	plt.plot(x1, y1)
	plt.yscale('linear')
	if (not net1_auto_yscale):
		plt.ylim(25000,30000)
	plt.title('Dual net losses', fontsize=10)
	plt.xlabel('number of non-idling iterations', fontsize=10)
	plt.ylabel('net1 loss', fontsize=10)
	plt.grid(True)

	# Plotting the output of net2
	plt.subplot(312)
	plt.plot(x2, y2)
	plt.yscale('linear')
	if (not net2_auto_yscale):
		plt.ylim(0.0,3.0)
	plt.xlabel('number of non-idling iterations', fontsize=10)
	plt.ylabel('net2 loss', fontsize=10)
	plt.grid(True)

if PLOT_SINGLE and single_path_exists:

	y3 = np.load("../../snapshots/net12_losses.npy")

	# Calculating the range of the function to plot, can be overwritten
	n3 = non_zero_entries(y3)
	if (net12_full_range):
		n3 = no_seqs*net2_seq_size
	x3 = range(n3)
	y3 = y3[0:n3]

	# Plotting the output of net12
	plt.subplot(313)
	plt.plot(x3, y3)
	plt.yscale('linear')
	if (not net12_auto_yscale):
		plt.ylim(0.5, 3)
	plt.title('Single net losses', fontsize=10)
	plt.xlabel('number of iterations', fontsize=10)
	plt.ylabel('net12 loss', fontsize=10)
	plt.grid(True)

os.system('mkdir ../../images')
plt.savefig('../../images/losses.png', fontsize=10)
plt.close()
