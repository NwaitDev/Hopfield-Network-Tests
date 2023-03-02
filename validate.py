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
	path1 = "img-data/L8x8.jpg"
	path2 = "img-data/O8x8.jpg"
	img1 = itm.importToMatrix(path1)
	img2 = itm.importToMatrix(path2)

	partialImg = np.zeros((8,8))
	for i in range(int(len(img1)/2)) :
		partialImg[i] = img1[i]
	
	printThatMatrix(partialImg, "PARTIAL")
	
	network = hn.networkFromImages([img1],imgHeight=8,imgWidth=8)
	printThatMatrix(network)
	partialImg = np.reshape(partialImg, (8*8,))
	for i in range(100):
		partialImg = hn.applyNetwork(partialImg,network)
	printThatMatrix(np.reshape(partialImg,(8,8)))
	