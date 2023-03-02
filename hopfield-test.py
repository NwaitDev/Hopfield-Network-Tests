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

def networkFromImages(imgSet, imgSize=64*64):
	"""
	Creates a matrix representing the wheights of 
	the edges between nodes of the network that stores the 
	image data from the imgSet
	TODO: no implementation yet
	"""
	return np.zeros((imgSize, imgSize))

def applyNetwork(inputImg, networkMatrix):
	"""
	Computes one iteration of the application 
	of the hopfield network to the input image 
	this should get a result closer to one 
	of the images stored in the network
	TODO: no implementation yet
	"""
	return np.zeros(imgSize)

def retrieveImage(inputImg, networkMatrix, synchronous=false):
	"""
	Computes an image that is stored in the networkMatrix
	based on the input image. By default, it is updating the values of randomly, one by one.
	TODO: no implementation yet
	"""
	return np.zeros(imgSize)