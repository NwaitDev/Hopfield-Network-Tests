import numpy as np
import src.imageToMatrix as itm
import random as rd

"""
We need :
	- a matrix that represents the network (if retreiving N*N images : it should be a (N²,N²) matrix)
	such matrix is related to the mean of the images stored in the matrix
	- a function that takes an image (/a key) and applies it the matrix calculus 
	to produce the next step to the expected result (it can have two modes : synchronous and asynchonous)
	- a function that iterates over the previous one to compute, step by step the target final image
	- an interface function that allows to use the functionnalities of this API
"""

def networkFromImages(imgSet, imgWidth=64, imgHeight=64):
	"""
	imgSet : array containing int matrices of shape (imgWidth,imgHeight)
	imgWidth, imgHeight : dimensions of the matrix representing the image
	Creates a matrix representing the wheights of 
	the edges between nodes of the network that stores the 
	images from the imgSet
	TODO: optimize it ?
	"""
	assert len(imgSet) != 0
	assert np.all(np.array([np.shape(x)==(imgWidth,imgHeight) for x in imgSet]))

	network = np.zeros((imgWidth*imgHeight, imgWidth*imgHeight))
	newShape = imgWidth*imgHeight
	for arr in imgSet:
		arr = np.resize(arr,newShape).astype('int')
		for i in range(newShape):
			for j in range(newShape):
				if(i!=j):
					network[i][j]+=arr[i]*arr[j]
	length = len(imgSet)
	mean_v = np.vectorize(lambda x : x/length)
	network = mean_v(network) 
	return network

def applyNetwork(inputImg, networkMatrix, synchronous=False):
	"""
	Computes one iteration of the application 
	of the hopfield network to the input image 
	this should get a result closer to one 
	of the images stored in the network
	TODO: no implementation yet
	"""
	
	if (synchronous):
		return np.product(inputImg,networkMatrix)
	outputImg = np.copy(inputImg)
	i = rd.randint(0, np.shape(inputImg)[0]-1)
	outputImg[i] = np.dot(inputImg, networkMatrix[i])
	return outputImg

def retrieveImage(inputImg, networkMatrix, synchronous=False):
	"""
	Computes an image that is stored in the networkMatrix
	based on the input image. By default, it is updating the values of randomly, one by one.
	TODO: no implementation yet
	"""
	return np.zeros(imgSize)