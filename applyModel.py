import numpy as np
from src.pltlib import printThatMatrix, PrintMatricesInGrid
import src.hopfieldnet as hn
import src.imageToMatrix as itm

if __name__ == "__main__":
	imgToRetrieve = itm.importToMatrix("./img-data/eye64x64.jpg")
	#partialImg = randomizeMatrix(imgToRetrieve.shape)
	partialImg = np.zeros(imgToRetrieve.shape)
	for i in range(int(len(imgToRetrieve)/2)) :
		partialImg[i] = imgToRetrieve[i]
	#printThatMatrix(partialImg, "PARTIAL")
	
	steps = hn.retrieveImage('./models/network.pk',partialImg, iterations=30000, stepsToPrint=15)
	PrintMatricesInGrid(steps)