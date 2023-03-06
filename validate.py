from src.testThatShit import test, printThatMatrix
import numpy as np
import hopfieldnet as hn
import src.imageToMatrix as itm
import pickle as pk


def basicTestSmallImage():
	#should fail once we replace the zeros by minus ones
	img = [[0,1],[0,1]]
	expected = np.array([[0,-1,-1,-1],[-1,0,-1,1],[-1,-1,0,-1],[-1,1,-1,0]])
	network = hn.networkFromImages([img],imgHeight=2,imgWidth=2)
	
	test("test Small image",network,expected)
	printThatMatrix(img)
	printThatMatrix(network)
	img2 = [0,0,0,0]
	for i in range(10):
		img2 = hn.applyNetwork(img2, network)
		printThatMatrix(np.reshape(img2,(2,2)))
	
def trainAndDumpNetwork():
	path1 = "img-data/eye64x64.jpg"
	path2 = "img-data/smile64x64.jpg"
	img1 = itm.importToMatrix(path1)
	img2 = itm.importToMatrix(path2)

	network = hn.networkFromImages([img1,img2],imgHeight=img1.shape[0],imgWidth=img1.shape[1])
	printThatMatrix(network, "NETWORK")
	hn.dumpNetwork(network,"network.pk")

def retrieveImage(partialImg):
	shape = partialImg.shape
	network = None
	with open("network.pk", "rb") as f:
		network = pk.load(f)

	partialImg = np.reshape(partialImg, (shape[0]*shape[1],))
	for i in range(50000):
		partialImg = hn.applyNetwork(partialImg,network)
		if(i%2000==0): 
			printThatMatrix(np.reshape(partialImg,(shape[0],shape[1])))

def randomizeMatrix(shape):
	return hn.zeroToMinusOne_v(np.random.randint(0,2,shape))

if __name__ == "__main__":
	
	#trainAndDumpNetwork()
#'''
	imgToRetrieve = itm.importToMatrix("img-data/eye64x64.jpg")
	#partialImg = randomizeMatrix(imgToRetrieve.shape)
	partialImg = np.zeros(imgToRetrieve.shape)
	for i in range(int(len(imgToRetrieve)/2)) :
		partialImg[i] = imgToRetrieve[i]
	printThatMatrix(partialImg, "PARTIAL")
	
	retrieveImage(partialImg)
#'''