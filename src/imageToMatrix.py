from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())  

img = Image.open(r"img-data/smile64x64.jpg")
arr = np.array(img)

def rgbToGreyScale(rgbArray):
	return (int(rgbArray[0])+int(rgbArray[1])+int(rgbArray[2]))/(3.0*255)

def rgbMatrixToGreyScale(rgbMatrix):
	arr = []
	for i in range(len(rgbMatrix)):
		for j in range(len(rgbMatrix[i])):
			arr.append(rgbToGreyScale(rgbMatrix[i][j]))
	return np.resize(np.array(arr),(64,64))

arr = rgbMatrixToGreyScale(arr)

plt.figure()
plt.imshow(arr)
plt.show()
