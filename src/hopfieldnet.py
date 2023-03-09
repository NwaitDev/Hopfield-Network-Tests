import numpy as np
import random as rd
import pickle as pk
import src.imageToMatrix as itm
from src.pltlib import printThatMatrix, PrintMatricesInGrid

"""
We need :
	- a matrix that represents the network (if retreiving N*N images : it should be a (N²,N²) matrix)
	such matrix is related to the mean of the images stored in the matrix
	- a function that takes an image (/a key) and applies it the matrix calculus 
	to produce the next step to the expected result (it can have two modes : synchronous and asynchonous)
	- a function that iterates over the previous one to compute, step by step the target final image
	- an interface function that allows to use the functionnalities of this API
"""

zeroToMinusOne_v = np.vectorize(lambda x : 2*x - 1 ) # replaces 0 by -1
minusOneToZero_v = np.vectorize(lambda x : (x+1)/2 ) # does the opposite
def threshold(x):
	if (x>0):
		return 1
	if (x<0):
		return -1
	return 0

thresholding_v = np.vectorize(threshold)

def trainAndDumpNetwork():
	path1 = "./img-data/simpsons/bart.png"
	path2 = "./img-data/simpsons/homer.png"
	path3 = "./img-data/simpsons/lisa.png"
	img1 = itm.importToMatrix(path1)
	img2 = itm.importToMatrix(path2)
	img3 = itm.importToMatrix(path3)

	network = networkFromImages([img1,img2,img3],imgHeight=img1.shape[0],imgWidth=img1.shape[1])
	printThatMatrix(network, "NETWORK")
	dumpNetwork(network,"network.pk")
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

def applyNetwork(inputImg, networkMatrix, synchronous=False):
	"""
	Computes one iteration of the application 
	of the hopfield network to the input image 
	this should get a result closer to one 
	of the images stored in the network
	"""
	if (synchronous):
		#return the thresholding of the product of the input image and the network matrix
		return thresholding_v(np.product(inputImg,networkMatrix))
	#copy the input image to compare the next state with the last state to check for convergence
	outputImg = np.copy(inputImg)
	#choose a random pixel to update
	i = rd.randint(0, np.shape(inputImg)[0]-1)
	#update the pixel i
	outputImg[i] = thresholding_v(np.dot(inputImg, networkMatrix[i]))
	return outputImg

def retrieveImage(networkPath, partialImg, iterations = 20000, stepsToPrint = 8):
	"""
	apply the network loaded from networkPath on n iterations on the input image. 
	"""
	shape = partialImg.shape
	network = None
	with open(networkPath, "rb") as f:
		network = pk.load(f)

	iters = iterations
	matrices = []
	itersToSave = iters//stepsToPrint

	partialImg = np.reshape(partialImg, (shape[0]*shape[1],))
	for i in range(iters):
		partialImg = applyNetwork(partialImg,network)
		if(i%itersToSave==0): 
			matrices.append(np.reshape(partialImg,(shape[0],shape[1])))
	return matrices
	
