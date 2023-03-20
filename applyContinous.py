import numpy as np
import src.modern_hopfield as mhn
from src.lib import *
import src.imageToMatrix as itm

def retrieveOnSimpsons():
    imgToRetrieve = itm.importToMatrix("./img-data/continuous/homer.png", "grayscale")
    partialImg = getRandomCroppedImageContinuous(imgToRetrieve,1)
    #partialImg = getCroppedImageContinuous(imgToRetrieve,0.7)
    memories = mhn.loadMemoriesContinuous("./img-data/continuous", imgWidth=64, imgHeight=64)
    printThatMatrix(partialImg, "Partial image", gray=True)
    mhn.retrieveImageContinuous(partialImg, imgToRetrieve, memories, maxIter=1, minIter=1, stepsToPrint=9, dimension=64)

if __name__ == "__main__":
    retrieveOnSimpsons()