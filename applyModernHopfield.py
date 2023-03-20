import numpy as np
import src.modern_hopfield as mhn
from src.lib import *
import src.imageToMatrix as itm

def retrieveOnSimpsons():
    imgToRetrieve = itm.importToMatrix("./img-data/simpsons/abraham.png")
    partialImg = getRandomCroppedImage(imgToRetrieve,50)
    memories = mhn.loadMemories("./img-data/simpsons", imgWidth=64, imgHeight=64)
    printThatMatrix(partialImg, "Partial image")
    mhn.retrieveImage(partialImg, imgToRetrieve, memories, synchronous=True, maxIter=1, minIter=1, stepsToPrint=9, dimension=64)

if __name__ == "__main__":
    retrieveOnSimpsons()