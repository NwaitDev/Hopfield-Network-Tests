import numpy as np
import src.modern_hopfield as mhn
from src.lib import *
import src.imageToMatrix as itm

def differences(img1,img2):
    difference = 0.0
    for i in range(len(img1)):
        for j in range(len(img2[i])):
            difference+=np.abs(img1[i][j]-img2[i][j])
    print(difference)

def retrieveOnSimpsons():
    imgToRetrieve = itm.importToMatrix("./img-data/continuous/marge.png", "grayscale")
    partialImg = imgToRetrieve
    partialImg = getRandomCroppedImageContinuous(imgToRetrieve,10,3)
    partialImg = getCroppedImageContinuous(partialImg,1.5)
    printThatMatrix(partialImg, "Partial image", gray=True)
    differences(partialImg,imgToRetrieve)
    memories = mhn.loadMemoriesContinuous("./img-data/continuous", imgWidth=64, imgHeight=64)
    outputImg = mhn.retrieveImageContinuous(partialImg, memories, maxIter=3, minIter=1, stepsToPrint=9, dimension=64)
    #compare values in outputImg and partialImg
    differences(outputImg,imgToRetrieve)
    #printThatMatrix(equals, "equals", gray=True)
    PrintMatricesInGrid([outputImg, imgToRetrieve], gray=True)
    #exportImage(outputImg, "homer_retrieved.png")

def retrieveMnist():
    mnist = np.load("./img-data/mnist_continuous.npz")
    imgToRetrieve = np.array(mnist["imgs"][17])
    imgToRetrieve = np.reshape(imgToRetrieve, (28,28))
    printThatMatrix(imgToRetrieve, "mnist x test 0", gray=True)
    mnist.close()
    partialImg = getRandomCroppedImageContinuous(imgToRetrieve,1,8)
    #partialImg = getCroppedImageContinuous(partialImg,1.5)
    printThatMatrix(partialImg, "mnist x test 0", gray=True)
    memories = mhn.loadMemoriesContinuousMnist()
    outputImg = mhn.retrieveImageContinuous(partialImg, memories, maxIter=1, minIter=1, stepsToPrint=9, dimension=28)
    PrintMatricesInGrid([imgToRetrieve,partialImg,outputImg], gray=True)

if __name__ == "__main__":
    #retrieveOnSimpsons()
    retrieveMnist()