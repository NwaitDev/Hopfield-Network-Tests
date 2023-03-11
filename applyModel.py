import numpy as np
from src.lib import *
import src.hopfieldnet as hn
import src.imageToMatrix as itm

if __name__ == "__main__":
	imgToRetrieve = itm.importToMatrix("./img-data/8x8/O8x8.jpg")
	
	#partialImg = randomizeMatrix(imgToRetrieve.shape)
	
	partialImg = np.zeros(imgToRetrieve.shape, dtype=int)
	imgDivisor = 2
	for i in range(int(len(imgToRetrieve)/imgDivisor)) :
		partialImg[i] = imgToRetrieve[i]
	for i in range(int(len(imgToRetrieve)/imgDivisor)+1,len(imgToRetrieve)) :
		randomOrder = np.random.permutation(len(imgToRetrieve[0]))
		for j in randomOrder :
			partialImg[i][j] = zeroToMinusOne(np.random.randint(0,2))
	
	#printThatMatrix(partialImg, "PARTIAL")
	
	steps = hn.retrieveImage('./models/network_8x8.pk',partialImg, iterations=256, stepsToPrint=9, sync=False)
	PrintMatricesInGrid(steps)