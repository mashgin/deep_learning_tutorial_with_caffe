#!/usr/bin/python -u
import caffe
import algo

caffe.set_mode_cpu()
#caffe.set_mode_gpu()

'''
	Load the solver.
'''
solver = caffe.get_solver('models/solver.proto')
solver.random_seed = 0xCAFFE

n_iter = 1000
test_interval = 100

'''
	Main solver loop.

	Caffe solver has two types of embedded nets: training net and test net.
	solver.net => training net
	solver.test_nets[0] => test net
'''
for it in range(n_iter):
	print '\rIteration', it,

	'''
		Generate batch_size samples and input them into the training network.
		Move solver by one step.
	'''
	batch_labels, batch_images = algo.GetBatch(64)
	solver.net.blobs['data'].data[...] = batch_images
	solver.net.blobs['label'].data[...] = batch_labels
	solver.step(1)

	'''
		After test_interval iterations, run tests to display accuracy
	'''
	if it % test_interval == 0:
		print '\nTesting...\t',

		total, correct = 0, 0
		for test_it in range(100):
			'''
				Generate a single random sample at a time and input to test network.
			'''
			batch_labels, batch_images = algo.GetBatch(1)
			solver.test_nets[0].blobs['data'].data[...] = batch_images

			'''
				Check if network returned the correct answer
			'''
			solver.test_nets[0].forward()
			if solver.test_nets[0].blobs['loss'].data.argmax() == batch_labels[0,0]:
				correct += 1
			total += 1
        
    	if total == correct:
        	break

		print "Accuracy: {0:.2f}%".format(correct * 100.0/total)
		

solver.net.save("models/weights.proto")













