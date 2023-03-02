from src.testThatShit import test, printThatMatrix
import numpy as np
import hopfieldnet as hn
import src.imageToMatrix as itm


def basicTestSmallImage():
	img = [[0,1],[0,1]]
	expected = np.array([[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,1,0,0]])
	actual = hn.networkFromImages([img],imgHeight=2,imgWidth=2)
	
	test("test testing",actual,expected)
	printThatMatrix(img)

if __name__ == "__main__":
	path = "img-data/eye64x64.jpg"
	img = itm.importToMatrix(path)
	network = hn.networkFromImages([img],imgHeight=64,imgWidth=64)

	printThatMatrix(img)
	printThatMatrix(network)
