import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import src.imageToMatrix as itm

zeroToMinusOne = lambda x : 2*x - 1 # replaces 0 by -1
minusOneToZero = lambda x : (x+1)/2 # does the opposite
zeroToMinusOne_v = np.vectorize(zeroToMinusOne) # replaces 0 by -1
minusOneToZero_v = np.vectorize(minusOneToZero) # does the opposite

def threshold(x):
	if (x>0):
		return 1
	if (x<0):
		return -1
	return 0

thresholding_v = np.vectorize(threshold)

def importImagesFromFolder(folderPath, imgWidth=64, imgHeight=64):
	"""
	Imports all the images from a folder and returns them as a list of matrices
	"""
	images = []
	for filename in os.listdir(folderPath):
		if filename.endswith(".png") or filename.endswith(".jpg"):
			images.append(itm.importToMatrix(folderPath+"/"+filename))
	return images

def getCroppedImage(img, div=2):
	croppedImg = np.zeros(img.shape, dtype=int)
	for i in range(int(len(img)/div)) :
		croppedImg[i] = img[i]
	for i in range(int(len(img)/div),len(img)) :
		randomOrder = np.random.permutation(len(img[0]))
		for j in randomOrder :
			croppedImg[i][j] = zeroToMinusOne(np.random.randint(0,2))
	return croppedImg

def getRandomCroppedImage(img, percent=10):
	croppedImg = np.zeros(img.shape, dtype=int)
	for i in range(len(img)) :
		randomOrder = np.random.permutation(len(img[0]))
		for j in randomOrder :
			if np.random.randint(0,100) > percent:
				croppedImg[i][j] = zeroToMinusOne(np.random.randint(0,2))
			else:
				croppedImg[i][j] = img[i][j]
	return croppedImg

def printThatMatrix(matrix, title=None):
	cmap = ListedColormap(["black","white"])
	imgplot = plt.imshow(matrix,cmap=cmap)
	if title!=None:
		plt.title(title)
		plt.show()
	else:
		plt.show()

def PrintMatricesInGrid(matrices):

	mLen = len(matrices)
	mLenSqrt = np.sqrt(mLen)
	rows = int(mLenSqrt)
	if(mLenSqrt%1!=0):
		rows+=1
	cols = len(matrices)//rows
	if len(matrices)%rows!=0:
		cols+=1
	
	cmap = ListedColormap(["black","white"])
	
	fig, axes = plt.subplots(rows,cols,figsize=(10,10))

	if isinstance(axes, np.ndarray):
		list_axes = list(axes.flat)
	else:
		list_axes = [axes]

	for i in range(len(list_axes)):
		list_axes[i].axes.get_xaxis().set_visible(False)
		list_axes[i].axes.get_yaxis().set_visible(False)
	
	for i in range(len(matrices)):
		list_axes[i].imshow(matrices[i], cmap=cmap)
	
	fig.tight_layout()
	plt.show()