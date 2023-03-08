from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

#return a matrix of -1 and 1 with blackAndWhite
#greyscale is not supported as of now
def importToMatrix(path, convTo= 'blackAndWhite'):
	img = Image.open(path)
	if(convTo == 'greyscale'):
		img = img.convert('L')
	elif(convTo == 'blackAndWhite'):
		img = img.convert('1')
	arr = np.array(img)
	return np.vectorize(lambda x : 2*x - 1 )(arr)

def rgbToGreyScale(rgbArray):
	return (int(rgbArray[0])+int(rgbArray[1])+int(rgbArray[2]))/(3.0*255)

def rgbMatrixToGreyScale(rgbMatrix):
	arr = []
	for i in range(len(rgbMatrix)):
		for j in range(len(rgbMatrix[i])):
			arr.append(rgbToGreyScale(rgbMatrix[i][j]))
	return np.resize(np.array(arr),(64,64))