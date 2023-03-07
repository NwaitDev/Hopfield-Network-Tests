import numpy as np
import src.imageToMatrix as itm
import random as rd
import pickle as pk

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
flip_v = np.vectorize(lambda x : -x) # flips the sign of the input
def threshold(x):
	if (x>0):
		return 1
	if (x<0):
		return -1
	return 0

thresholding_v = np.vectorize(threshold)

def networkFromImages(imgSet, imgWidth=64, imgHeight=64, maxElementVal=1):
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
	with open("network.pk", "wb") as f:
		pk.dump(network, f)
  
  
def energyFunc(x: np.ndarray[np.float64, 1], imglist, func = np.exp):
	s = 0	
	for img in imglist:
		s += func(np.dot(x, img))
	return - s

'''
'''

def newPixelWithModernNetwork(inputImg: np.ndarray[np.float64, 2]):
	path1 = "img-data/eye64x64.jpg"
	path2 = "img-data/smile64x64.jpg"
	img1 = np.reshape(itm.importToMatrix(path1), (64*64, 1))
	img2 = np.reshape(itm.importToMatrix(path2), (64*64, 1))

	return threshold(-energyFunc(inputImg, [img1, img2])+ energyFunc(flip_v(inputImg), [img1, img2]))
    

def applyNetwork(inputImg : np.ndarray[np.float64], networkMatrix : np.ndarray[np.float64, 2], synchronous=False):
	"""
	Computes one iteration of the application
	of the hopfield network to the input image 
	this should get a result closer to one 
	of the images stored in the networkZ
	TODO: no implementation yet
	"""
	if (synchronous):
		#return the thresholding of the product of the input image and the network matrix
		return thresholding_v(np.product(inputImg,networkMatrix))
	#copy the input image to compare the next state with the last state to check for convergence
	outputImg = np.copy(inputImg)
	#choose a random pixel to update
	i = rd.randint(0, np.shape(inputImg)[0]-1)
	#update the pixel i
	outputImg[i] = newPixelWithModernNetwork(inputImg)
	return outputImg

def retrieveImage(inputImg, networkMatrix, synchronous=False):
	"""
	Computes an image that is stored in the networkMatrix
	based on the input image. By default, it is updating the values of randomly, one by one.
	TODO: no implementation yet
	"""
	return np.zeros(imgSize)