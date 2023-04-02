import numpy as np
from src.lib import *
import src.hopfieldnet as hn
import src.imageToMatrix as itm

def retrieveOnSimpsons():
	imgToRetrieve = itm.importToMatrix("./img-data/simpsons/homer.png")
	partialImg = getCroppedImage(imgToRetrieve,2)
	steps = hn.retrieveImage('./models/network_simpsons_allimgs.pk',partialImg, sync=False, iterations=25000, stepsToPrint=9)
	PrintMatricesInGrid(steps)

if __name__ == "__main__":
	retrieveOnSimpsons()