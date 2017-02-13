#!/usr/local/bin/python

'''
	Generate fake data.
'''

import numpy as np
import cv2
import sys, os
import random
import math

n_rows, n_cols, n_ch  = 100, 100, 1
n_classes = 2

def ShowImage(img):
	'''
		Since incoming image is normalized to 0-1 (float) range, 
		multiply it with 255 and convert to uint8 for display purposes
	'''
	img = np.array(img * 255, dtype='uint8').reshape(n_rows, n_cols, n_ch)
	cv2.imshow("Image", img)
	return cv2.waitKey(0)

def GetSample():
	'''
		Generate random class:
			0 = circle
			1 = square

		Draw circle / square of random size.
		Normalize output image by dividing by 255. Convert image to float32
	'''
	cls = random.randint(0, n_classes-1)
	img = np.zeros((n_rows, n_cols, n_ch), dtype='uint8')

	if cls == 0:
		cv2.circle(img, (n_cols/2, n_rows/2), random.randint(10, n_cols/2 - 10), (255), -1)
	elif cls == 1:
		side = random.randint(10, n_cols/2 - 10)
		cv2.rectangle(img, (n_cols/2 - side, n_cols/2 - side), (n_cols/2 + side, n_cols/2 + side), (255), -1)

	channels = cv2.split(img)
	out_image = np.array(channels, dtype='float32') / 255.

	return cls, out_image

def GetBatch(batch_size):
	'''
		Return an array of multiple samples
	'''
	batch_labels, batch_images = [], []

	for i in range(0, batch_size):
		cls, image = GetSample()
		batch_images.append(image)
		batch_labels.append(cls)

	return np.array(batch_labels).reshape(batch_size, 1), np.array(batch_images)

if __name__ == '__main__':
	'''
		Visualize samples being generated if this file is run directly
	'''
	batch_labels, batch_images = GetBatch(1)
	print batch_images.shape, batch_labels.shape

	while True:
		cls, img = GetSample()
		print cls
		if ShowImage(img) == 113:
			quit()






