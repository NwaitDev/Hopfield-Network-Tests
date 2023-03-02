import numpy as np
import src.imageToMatrix as itm


"""
We need :
	- a matrix that represents the network (if retreiving N*N images : it should be a (N²,N²) matrix)
	such matrix is related to the mean of the images stored in the matrix
	- a function that takes an image (/a key) and applies it the matrix calculus 
	to produce the next step to the expected result (it can have two modes : synchronous and asynchonous)
	- a function that iterates over the previous one to compute, step by step the target final image
	- an interface function that allows to use the functionnalities of this API
"""

def networkFromImages(imgSet, imgSize=64):
	"""
	imgSet : array containing int matrix of shape (imgSize,imgSize)
	imgSize : number of pixels in a line of the image
	Creates a matrix representing the wheights of 
	the edges between nodes of the network that stores the 
	image data from the imgSet
	TODO: no implementation yet
	"""
	assert len(imgSet) != 0
	network = np.zeros((imgSize**2, imgSize**2))
	for arr in imgSet:
		(rows, columns) = np.shape(arr)
		newShape = rows * columns
		arr = np.resize(arr,newShape)
		for i in range(newShape):
			for j in range(newShape):
				if(i!=j):
					network[i][j]+=int(arr[i])*int(arr[j])
	length = len(imgSet)
	mean_v = np.vectorize(lambda x : x/length)
	network = mean_v(network) 
	return network

def applyNetwork(inputImg, networkMatrix):
	"""
	Computes one iteration of the application 
	of the hopfield network to the input image 
	this should get a result closer to one 
	of the images stored in the network
	TODO: no implementation yet
	"""
	return np.zeros(imgSize)

def retrieveImage(inputImg, networkMatrix, synchronous=False):
	"""
	Computes an image that is stored in the networkMatrix
	based on the input image. By default, it is updating the values of randomly, one by one.
	TODO: no implementation yet
	"""
	return np.zeros(imgSize)