import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os.path

printDual = os.path.isfile("../../models/snapshots/net1_losses.npy")
printSingle = os.path.isfile("../../models/snapshots/net12_losses.npy")

import matplotlib
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

def nonZeroEntries(numbers):
	for i in range(numbers.size):
		if numbers[i]==0:
			break
	return i


if printDual:

	y1 = np.load("../../models/snapshots/net1_losses.npy")
	y2 = np.load("../../models/snapshots/net2_losses.npy")

	n1 = nonZeroEntries(y1)
	n1=1000
	x1 = range(n1)
	y1 = y1[0:n1]
	y2 = y2[0:n1]

	#x = range(y1.shape[0])

	plt.figure(1)

	plt.subplot(311)
	plt.plot(x1, y1)
	plt.yscale('linear')
	#plt.ylim(25000,30000)
	plt.title('Dual net losses', fontsize=10)
	plt.xlabel('number of iterations', fontsize=10)
	plt.ylabel('net1 loss', fontsize=10)
	plt.grid(True)


	plt.subplot(312)
	plt.plot(x1, y2)
	plt.yscale('linear')
	plt.ylim(0.5,3.0)
	plt.xlabel('number of iterations', fontsize=10)
	plt.ylabel('net2 loss', fontsize=10)
	plt.grid(True)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

if printSingle:

	y3 = np.load("../../models/snapshots/net12_losses.npy")

	n2 = nonZeroEntries(y3)
	x2 = range(n2)
	y3 = y3[0:n2]

	plt.subplot(313)
	plt.plot(x2, y3)
	plt.yscale('linear')
	#plt.ylim(0.5, 3)
	plt.title('Single net losses', fontsize=10)
	plt.xlabel('number of iterations', fontsize=10)
	plt.ylabel('net12 loss', fontsize=10)
	plt.grid(True)

# plt.show()
plt.savefig('../../images/losses.png', fontsize=10)
plt.close()
