import numpy as np
import src.modern_hopfield as mhn
from src.lib import *
import src.imageToMatrix as itm

if __name__ == "__main__":
    memories = mhn.loadMemoriesContinuous("./img-data/continuous", imgWidth=64, imgHeight=64)
    homer = itm.importToMatrix("./img-data/continuous/homer.png", "grayscale")
    marge = itm.importToMatrix("./img-data/continuous/marge.png", "grayscale")
    #printThatMatrix(homer, "input", gray=True)
    homer = np.reshape(homer, (len(homer[0])*len(homer[0]),))
    marge = np.reshape(marge, (len(marge[0])*len(marge[0]),))
    e1 = mhn.energyContinuous(homer,memories)
    e2 = mhn.energyContinuous(marge,memories)
    print(e1)
    print(e2)