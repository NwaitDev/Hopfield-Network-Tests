import numpy as np
from src.lib import *
import src.hopfieldnet as hn
import src.imageToMatrix as itm

if __name__ == "__main__":
	imgToRetrieve = itm.importToMatrix("./img-data/simpsons/bart.png")
	#partialImg = randomizeMatrix(imgToRetrieve.shape)
	partialImg = np.zeros(imgToRetrieve.shape, dtype=int)
	for i in range(int(len(imgToRetrieve)/3)) :
		partialImg[i] = imgToRetrieve[i]
	for i in range(int(len(imgToRetrieve)/3)+1,len(imgToRetrieve)) :
		randomOrder = np.random.permutation(len(imgToRetrieve[0]))
		for j in randomOrder :
			partialImg[i][j] = zeroToMinusOne(np.random.randint(0,2))
	#printThatMatrix(partialImg, "PARTIAL")
	
	steps = hn.retrieveImage('./models/network-simpsons.pk',partialImg, iterations=8, stepsToPrint=16, sync=True)
	PrintMatricesInGrid(steps)