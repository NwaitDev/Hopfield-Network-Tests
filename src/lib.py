import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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