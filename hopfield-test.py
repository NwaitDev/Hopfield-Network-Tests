import numpy as np
import src.imageToMatrix as itm

"""
We need :
	- a matrix that represents the network (if retreiving N*N images : it should be a (N²,N²) matrix)
	such matrix is related to the mean of the images stored in the matrix
	- a functions that takes an image (/a key) and applies it the matrix calculus 
	to produce the next step to the expected result (it can have two modes : synchronous and asynchonous)
	- a function that iterates over the previous one to compute, step by step the target final image
	- an interface function that allows to use the functionnalities of this API
"""

