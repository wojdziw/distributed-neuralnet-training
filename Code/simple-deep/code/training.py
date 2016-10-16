import caffe

solver = caffe.get_solver('../models/solver.prototxt')

maxIter = 10
stepPerIter = 1

for _ in range(maxIter):
	solver.step(1)

	# conv 1 parameters
	conv1Params = solver.net.params['conv1'][0].data

	# communicate this somewhere - to the other net
