import numpy as np
import src.modern_hopfield as mhn
from src.lib import *
import src.imageToMatrix as itm

def retrieveOnSimpsons():
    imgToRetrieve = itm.importToMatrix("./img-data/continuous/marge.png", "grayscale")
    partialImg = getRandomCroppedImageContinuous(imgToRetrieve,10,3)
    #partialImg = getCroppedImageContinuous(partialImg,1.5)
    printThatMatrix(partialImg, "Partial image", gray=True)
    memories = mhn.loadMemoriesContinuous("./img-data/continuous", imgWidth=64, imgHeight=64)
    outputImg = mhn.retrieveImageContinuous(partialImg, imgToRetrieve, memories, maxIter=1, minIter=1, stepsToPrint=9, dimension=64)
    #compare values in outputImg and partialImg
    areEquals = 0
    areNotEquals = 0.0
    equals = np.zeros(imgToRetrieve.shape)
    for i in range(len(outputImg)):
        for j in range(len(outputImg[i])):
            if(outputImg[i][j] == partialImg[i][j]):
                areEquals += 1
                equals[i][j] = outputImg[i][j]
            else:
                areNotEquals+=np.abs(outputImg[i][j]-partialImg[i][j])
    print(areEquals)
    print(areNotEquals)
    print(areNotEquals/(imgToRetrieve.shape[0]*imgToRetrieve.shape[0]-areEquals))
    #printThatMatrix(equals, "equals", gray=True)
    #PrintMatricesInGrid([outputImg, imgToRetrieve], gray=True)
    #exportImage(outputImg, "homer_retrieved.png")

if __name__ == "__main__":
    retrieveOnSimpsons()