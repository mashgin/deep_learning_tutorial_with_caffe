#!/usr/bin/env python

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import sys, os

n_classes = 2
n_classifiers = 1
n_rows, n_cols = 100, 100

def GetNetwork(mode, batch_size):
	'''
		Network definition
	'''
	n = caffe.NetSpec()

	n.data = L.Input(input_param={'shape': {'dim': [batch_size, 1, n_rows, n_cols]}})
	if mode == 'train':
		n.label = L.Input(input_param={'shape': {'dim': [batch_size, n_classifiers]}})

	n.conv1 = L.Convolution(n.data, kernel_size=5, stride=1, num_output=16, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.conv1, in_place=True)
	n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=3, pool=P.Pooling.MAX)
	n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

	n.fc1 = L.InnerProduct(n.norm1, num_output=64, weight_filler=dict(type='xavier'))
	n.fcrelu1 = L.ReLU(n.fc1, in_place=True)
	n.drop1 = L.Dropout(n.fcrelu1, dropout_ratio=0.25, in_place=True)

	n.score_all = L.InnerProduct(n.drop1, num_output=n_classes * n_classifiers, weight_filler=dict(type='xavier'))
	n.score = L.Reshape(n.score_all, reshape_param={'shape': {'dim': [batch_size, n_classes, n_classifiers]}})

	if mode == 'train':
		n.loss = L.SoftmaxWithLoss(n.score, n.label)
	else:
		n.loss = L.Softmax(n.score)

	return str(n.to_proto())

def GetSolver():
	'''
		Solver definition
	'''
	s = caffe_pb2.SolverParameter()

	s.train_net = 'models/train.proto'
	s.test_net.append('models/test.proto')
	s.test_interval = 1000000
	s.test_iter.append(10)
	s.max_iter = 1000000
	s.display = 1000000

	s.type = "Adam"
	s.base_lr = 0.001
	s.momentum = 0.9
	s.momentum2 = 0.999
	s.lr_policy = 'fixed'

	return s

def WriteModel():
	'''
		Generate train, test and deploy nets with different params. Save them to file.
		deploy == test
	'''
	with open('models/train.proto', 'w') as f:
		f.write(GetNetwork('train', 64))
	with open('models/test.proto', 'w') as f:
		f.write(GetNetwork('test', 1))
	with open('models/deploy.proto', 'w') as f:
		f.write(GetNetwork('deploy', 1))

def WriteSolver():
	'''
		Generate and save the solver params
	'''
	with open('models/solver.proto', 'w') as f:
		f.write(str(GetSolver()))

def ShowNet():
	'''
		Load the deploy net and display its blob sizes
	'''
	net = caffe.Net('models/deploy.proto', caffe.TEST)
	for k in net.blobs.keys():
		print k, net.blobs[k].data.shape

if __name__ == '__main__':

	if not os.path.isdir('models'):
		os.mkdir('models')

	WriteModel()
	WriteSolver()

	print "================ Network topology ================"
	ShowNet()
	print "=================================================="



# n.loss = L.SoftmaxWithLoss(n.score, n.label)
# n.loss = L.Softmax(n.score)
# n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
# n.loss = L.Sigmoid(n.score)













