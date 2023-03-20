import numpy as np
import random as rd
from src.lib import *

def loadMemories(path, imgWidth=64, imgHeight=64):
    memories = np.array(importImagesFromFolder(path, imgWidth, imgHeight))
    memories =  memories.reshape(len(memories), imgWidth*imgHeight)
    return memories

def loadMemoriesContinuous(path, imgWidth=64, imgHeight=64):
    memories = np.array(importImagesFromFolder(path, imgWidth, imgHeight, imgConvType='grayscale'))
    memories = memories.reshape(len(memories), imgWidth*imgHeight)
    return memories

def retrieveImageContinuous(inputImg, objImg, memories, maxIter=8, minIter=2, stepsToPrint=9, dimension=64):
    """
    Retrieves the image from the memories
    """
    inputImg = np.reshape(inputImg, (len(inputImg[0])*len(inputImg[0]),))
    outputImg = np.copy(inputImg)
    
    for i in range(maxIter):
        outputImg = updateSyncContinuous(outputImg, memories, 1)
    printThatMatrix(np.reshape(outputImg, (dimension,dimension)), "Output image", gray=True)
    return outputImg

def retrieveImage(inputImg, objImg, memories, synchronous=True, maxIter=8, minIter=2, stepsToPrint=9, dimension=64):
    """
    Retrieves the image from the memories
    """
    inputImg = np.reshape(inputImg, (len(inputImg[0])*len(inputImg[0]),))
    outputImg = np.copy(inputImg)
    
    for i in range(maxIter):
        if(synchronous):
            outputImg = updateSync(outputImg, memories, energyWithLSE, 4)
        else:
            i = rd.randint(0, np.shape(inputImg)[0]-1)
            outputImg = updateAsync(outputImg, memories,i, energyWithLSE, 4)
        if(i>minIter and np.array_equal(outputImg, objImg)):
            return outputImg
    printThatMatrix(np.reshape(outputImg, (dimension,dimension)), "Output image")
    return outputImg

#fig9
def energy(img, memories, B=1):
    #return -np.exp(np.log(np.sum((np.tile(img, (len(memories), 1)) * memories)))).sum()
    #x = (np.dot(memories.T,np.tile(img, (len(memories), 1))))
    #x = np.matmul(memories,img.T) smarter but overflow
    N = len(memories)
    x = np.zeros(N)
    for i in range(N):
        x[i] = np.dot(memories[i],img)
    return - F(x, 5).sum()

def energyWithLSE(img, memories, B=1, reducer=1):
    N = len(memories)
    x = np.zeros(N)
    reducer = 0.02 
    #for i in range(N):
    #    while(np.exp(np.dot(memories[i],img) * reducer) > 10e+40):
    #        reducer /= 2
    for i in range(N):
        x[i] = np.dot(memories[i],img) * reducer
    return - np.exp(lse(B,x))

def energyContinuous(img, memories, B=1):
    N = len(memories)
    x = np.zeros(N)
    for i in range(N):
        x[i] = np.dot(memories[i],img)
    racooncav = - lse(B,x)
    M = 0
    for i in range(N):
        M_temp = np.ones(img.shape[0]) @ memories[i]
        if(M_temp > M):
            M = M_temp
    racoonvex = 0.5 * (img.dot(img) + np.power(M,2)) + 1/B * np.log(N)
    return racooncav + racoonvex

def lse(B,x):
    return 1/B * np.log(np.sum(np.exp(B*x)))

def F(x, n):
    '''Rectified polynomial'''
    x[x < 0] = 0
    return np.power(x, n)

def updateSyncContinuous(inputImg, memories, B=1):
    """
	Updates the image according to the energy function
	"""
    outputImg = np.copy(inputImg)
    randomOrder = np.random.permutation(len(inputImg))
    for i in randomOrder:
        newState = memories.T @ softmax(B * memories @ outputImg)
        outputImg = newState
    return outputImg

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def updateSync(inputImg, memories, func=energy, B=1):
    """
	Updates the image according to the energy function
	"""
    outputImg = np.copy(inputImg)
    randomOrder = np.random.permutation(len(inputImg))
    for i in randomOrder:
        newImage = np.copy(outputImg)
        newImage[i] *= -1
        newEnergy = func(newImage, memories, B)
        oldEnergy = func(outputImg, memories, B)
        energyDiff = newEnergy - oldEnergy
        if(energyDiff < 0):
            outputImg[i] *= -1
    return outputImg

def updateAsync(inputImg, memories, index, func=energy, B=1):
    """
    Updates the image according to the energy function
    """
    outputImg = np.copy(inputImg)
    newImage = np.copy(outputImg)
    newImage[index] *= -1
    newEnergy = func(newImage, memories, B)
    oldEnergy = func(outputImg, memories, B)
    energyDiff = newEnergy - oldEnergy
    if(energyDiff < 0):
        outputImg[index] *= -1
    return outputImg