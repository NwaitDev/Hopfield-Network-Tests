from src.testThatShit import test
import numpy as np
import hopfieldnet as hn
import src.imageToMatrix as itm

path = "img-data/eye64x64.jpg"

img = itm.importToMatrix(path,convTo='blackAndWhite')

test("test test", hn.networkFromImages([img]))
