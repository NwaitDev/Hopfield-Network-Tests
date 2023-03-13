import numpy as np
import random as rd
from src.lib import *

def loadMemories(path, imgWidth=64, imgHeight=64):
    memories = np.array(importImagesFromFolder(path, imgWidth, imgHeight))
    memories =  memories.reshape(len(memories), imgWidth*imgHeight)
    return memories

def retrieveImage(inputImg, objImg, memories, synchronous=True, maxIter=8, minIter=2, stepsToPrint=9):
    """
    Retrieves the image from the memories
    """
    inputImg = np.reshape(inputImg, (len(inputImg[0])*len(inputImg[0]),))
    outputImg = np.copy(inputImg)
    
    for i in range(maxIter):
        if(synchronous):
            outputImg = updateSync(outputImg, memories)
        else:
            i = rd.randint(0, np.shape(inputImg)[0]-1)
            outputImg = updateAsync(outputImg, memories,i)
        if(i>minIter and np.array_equal(outputImg, objImg)):
            return outputImg
    printThatMatrix(np.reshape(outputImg, (64,64)), "Output image")
    print("Did not converge after "+str(maxIter)+" iterations")
    return outputImg

#fig9
def energy(img, memories):
    x = (np.tile(img, (len(memories), 1)) * memories).sum(1)
    return - F(x, 2.8).sum()

def F(x, n):
    '''Rectified polynomial'''
    x[x < 0] = 0.
    return x**n*20

def updateSync(inputImg, memories):
    """
	Updates the image according to the energy function
	"""
    outputImg = np.copy(inputImg)
    randomOrder = np.random.permutation(len(inputImg))
    for i in randomOrder:
        newImage = np.copy(outputImg)
        newImage[i] *= -1
        energyDiff = energy(newImage, memories) - energy(outputImg, memories)
        if(energyDiff < 0):
            outputImg[i] *= -1
    printThatMatrix(np.reshape(outputImg, (64,64)), "Output image")
    return outputImg

def updateAsync(inputImg, memories, index):
    """
    Updates the image according to the energy function
    """
    outputImg = np.copy(inputImg)
    newImage = np.copy(outputImg)
    newImage[index] *= -1
    energyDiff = energy(newImage, memories) - energy(outputImg, memories)
    if(energyDiff < 0):
        outputImg[index] *= -1
    return outputImg