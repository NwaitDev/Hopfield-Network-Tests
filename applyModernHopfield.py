import numpy as np
import src.modern_hopfield as mhn
from src.lib import *
import src.imageToMatrix as itm

def retrieveOnSimpsons():
    imgToRetrieve = itm.importToMatrix("./img-data/simpsons/bart.png")
    partialImg = getCroppedImage(imgToRetrieve,2)
    memories = mhn.loadMemories("./img-data/simpsons", imgWidth=64, imgHeight=64)
    mhn.retrieveImage(partialImg, imgToRetrieve, memories, synchronous=True, maxIter=2, minIter=1, stepsToPrint=9)

if __name__ == "__main__":
    retrieveOnSimpsons()