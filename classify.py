#!/usr/bin/python -u

'''
	Run: "./classify.py" to get accuracy numbers.
	Run: "./classify.py 1" to see results one by one.
'''
# import numpy as np
import caffe
# import cv2
import algo
import sys

caffe.set_mode_cpu()
#caffe.set_mode_gpu()

'''
	Load the deploy net along with trained weights
'''
net = caffe.Net('models/deploy.proto', 'models/weights.proto', caffe.TEST)

total, correct = 0, 0
n_iter = 1000

'''
	Run test for lots of samples to get accuracy numbers
'''
for test_it in range(n_iter):
	if len(sys.argv) == 1 and test_it % 10 == 0:
		print "\rIter:", test_it,

	'''
		Generate a random sample and input to network
	'''
	batch_labels, batch_images = algo.GetBatch(1)
	net.blobs['data'].data[...] = batch_images

	'''
		net.forward() to compute the results
	'''
	net.forward()


	'''
		Check if answer was correct
	'''
	if net.blobs['loss'].data.argmax() == batch_labels[0,0]:
		correct += 1
	total += 1

	'''
		Print and display results
	'''
	if len(sys.argv) == 2:
		if net.blobs['loss'].data.argmax() == 0:
			print "Circle"
		else:
			print "Square"
		if algo.ShowImage(batch_images[0]) == 113:
			quit()

print "\nAccuracy: {0:.2f}%".format(correct * 100.0/total)




















