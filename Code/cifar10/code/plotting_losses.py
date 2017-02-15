import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os.path

printDual = os.path.isfile("../models/snapshots/net1_losses.npy")
printSingle = os.path.isfile("../models/snapshots/net12_losses.npy")

import matplotlib
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

def nonZeroEntries(numbers):
	for i in range(numbers.size):
		if numbers[i]==0:
			break
	return i


if printDual:

	y1 = np.load("../models/snapshots/net1_losses.npy")
	y2 = np.load("../models/snapshots/net2_losses.npy")

	n = nonZeroEntries(y1)
	x = range(n)
	y1 = y1[0:n]
	y2 = y2[0:n]

	#x = range(y1.shape[0])

	plt.figure(1)

	plt.subplot(211)
	plt.plot(x, y1)
	plt.yscale('log')
	#plt.ylim(25000,30000)
	plt.title('Dual net losses', fontsize=10)
	plt.xlabel('number of iterations', fontsize=10)
	plt.ylabel('net1 loss', fontsize=10)
	plt.grid(True)


	plt.subplot(212)
	plt.plot(x, y2)
	plt.yscale('linear')
	#plt.ylim(80,90)
	plt.xlabel('number of iterations', fontsize=10)
	plt.ylabel('net2 loss', fontsize=10)
	plt.grid(True)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

# if printSingle:
#
# 	y3 = np.load("../models/snapshots/net12_losses.npy")
#
# 	x = range(y3.shape[0])
#
# 	plt.subplot(313)
# 	plt.plot(x, y3)
# 	plt.yscale('linear')
# 	plt.ylim(0.5, 3)
# 	plt.title('Net12 losses', fontsize=10)
# 	plt.grid(True)

# plt.show()
plt.savefig('../images/losses.png', fontsize=10)
plt.close()
