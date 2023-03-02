from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def importToMatrix(path, convTo= 'greyscale'):
	img = Image.open(path)
	if(convTo == 'greyscale'):
		img = img.convert('L')
	elif(convTo == 'blackAndWhite'):
		img = img.convert('1')
	arr = np.array(img)
	return arr

def rgbToGreyScale(rgbArray):
	return (int(rgbArray[0])+int(rgbArray[1])+int(rgbArray[2]))/(3.0*255)

def rgbMatrixToGreyScale(rgbMatrix):
	arr = []
	for i in range(len(rgbMatrix)):
		for j in range(len(rgbMatrix[i])):
			arr.append(rgbToGreyScale(rgbMatrix[i][j]))
	return np.resize(np.array(arr),(64,64))

if __name__ == "__main__":
	print(os.getcwd())
	arr = importToMatrix('./img-data/eye64x64.jpg', 'blackAndWhite')
	imgplot = plt.imshow(arr)
	plt.show()
