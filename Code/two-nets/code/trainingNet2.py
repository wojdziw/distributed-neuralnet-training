import caffe

solver = caffe.get_solver('../models/solverNet2.prototxt')

maxIter = 1
stepPerIter = 1

for _ in range(maxIter):
	solver.step(1)

	# data computed by each of the layers
	originalData = solver.net.blobs['data'].data
	dataConv1 = solver.net.blobs['conv1'].data
	dataPool1 = solver.net.blobs['pool1'].data
	dataConv2 = solver.net.blobs['conv2'].data
	dataPool2 = solver.net.blobs['pool2'].data
	dataConv3 = solver.net.blobs['conv3'].data
        dataPool3 = solver.net.blobs['pool3'].data 
	
	# parameters computed by each of the layers
	paramsConv1 = solver.net.params['conv1'][0].data
	paramsConv2 = solver.net.params['conv2'][0].data
	paramsConv3 = solver.net.params['conv3'][0].data
	
	# save the data produced by conv2 and give it to the other net

	# the new data gets communicated and processed by the other net
	# what should be the target labels for this net beeeee	

	# after the other net is forward and back propped:
	# RUN ADMM
