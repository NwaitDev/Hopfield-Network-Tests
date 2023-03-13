import numpy as np
from src.lib import *
import src.hopfieldnet as hn
import src.imageToMatrix as itm

def retrieveOnSimpsons():
	imgToRetrieve = itm.importToMatrix("./img-data/simpsons/abraham.png")
	partialImg = getCroppedImage(imgToRetrieve,2)
	steps = hn.retrieveImage('./models/network_simpsons_allimgs.pk',partialImg, sync=False, iterations=2000, stepsToPrint=9)
	PrintMatricesInGrid(steps)

if __name__ == "__main__":
	retrieveOnSimpsons()