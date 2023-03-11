import numpy as np
from src.lib import printThatMatrix
import hopfieldnet as hn
from colorama import Fore, Style

def failMessage(testName, actual, expected):
	print(Fore.RED + "[✗] : %s\nactual :\n%s\nexpected :\n%s"%(testName,str(actual),str(expected))+Style.RESET_ALL)

def passMessage(testName):
	print(Fore.GREEN + "[✔] : %s"%(testName)+Style.RESET_ALL)

def test(testName, actual, expected=True):
	if(type(actual)!=type(expected)):
		failMessage(testName, actual, expected)
		return
	if(type(actual)==np.ndarray):
		if(type(expected)==np.ndarray):
			if(np.shape(actual)==np.shape(expected)):
				if(np.all(actual==expected)):
					passMessage(testName)
					return
				print("different values among arrays")
				failMessage(testName, actual, expected)
				return
			print("trying to compare arrays of different shapes")
			failMessage(testName, actual, expected)
			return
		print("trying to compare data of different types...")
		failMessage(testName, actual, expected)
		return
	if(actual==expected):
		passMessage(testName)
		return
	failMessage(testName, actual, expected)


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

if __name__ == "__main__":
	test("equality", 1,1)
	test("equality2",1,2)
	test("equality3",1==1)
	test("equality4",np.array(np.linspace(0,9,10)),np.array(np.linspace(0,9,10)))
	test("equality5",np.array(np.linspace(0,9,10)),np.array(np.linspace(0,9,11)))