import numpy as np
import random as rd
import pickle as pk
from src.lib import *

"""
We need :
	- a matrix that represents the network (if retreiving N*N images : it should be a (N²,N²) matrix)
	such matrix is related to the mean of the images stored in the matrix
	- a function that takes an image (/a key) and applies it the matrix calculus 
	to produce the next step to the expected result (it can have two modes : synchronous and asynchonous)
	- a function that iterates over the previous one to compute, step by step the target final image
	- an interface function that allows to use the functionnalities of this API
"""

def trainAndDumpNetwork(imgs, modelName="network_unnamed"):

	if (imgs is None):
		return None

	network = networkFromImages(imgs,imgHeight=imgs[0].shape[0],imgWidth=imgs[0].shape[1])
	
	dumpPath = "./models/"+modelName
	if(dumpPath[-3:]!=".pk"):
		dumpPath+=".pk"
	dumpNetwork(network, dumpPath)
	return network

def randomizeMatrix(shape):
	return zeroToMinusOne_v(np.random.randint(0,2,shape))

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
	network = thresholding_v(network)
	return network

def dumpNetwork(network, path):
	"""
	Dumps the network matrix to a file
	"""
	with open(path, "wb") as f:
		pk.dump(network, f)

def loadNetwork(path):
	"""
	Loads a network matrix from a file
	"""
	with open(path, "rb") as f:
		return pk.load(f)

def applyNetworkDiscrete(inputImg, networkMatrix, synchronous=False):
	"""
	Computes one iteration of the application 
	of the hopfield network to the input image 
	this should get a result closer to one 
	of the images stored in the network
	"""
	if (synchronous):
		#return the thresholding of the product of the input image and the network matrix
		return thresholding_v(np.dot(networkMatrix,inputImg))
	
	outputImg = np.copy(inputImg)
	#choose a random pixel to update
	i = rd.randint(0, np.shape(inputImg)[0]-1)
	#update the pixel i
	outputImg[i] = thresholding_v(np.dot(inputImg, networkMatrix[i]))
	return outputImg

def retrieveImage(networkPath, partialImg, sync=False, iterations = 20000, stepsToPrint = 8):
	"""
	apply the network loaded from networkPath on n iterations on the input image. 
	"""
	shape = partialImg.shape
	network = loadNetwork(networkPath)

	if(stepsToPrint>iterations):
		stepsToPrint = iterations

	print("shapes : ",shape, " ", network.shape)

	iters = iterations
	matrices = []
	itersToSave = iters//(stepsToPrint)

	matrices.append(partialImg)

	update = applyNetworkDiscrete

	partialImg = np.reshape(partialImg, (shape[0]*shape[1],))
	for i in range(iters):
		partialImg = update(partialImg,network,synchronous=sync)
		if(i%itersToSave==0 or i==iters-1):
			if(len(matrices)==stepsToPrint):
				matrices[stepsToPrint-1] = np.reshape(partialImg,shape)
			else:
				matrices.append(np.reshape(partialImg,shape))
	return matrices
	
