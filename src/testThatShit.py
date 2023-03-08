import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from matplotlib.colors import ListedColormap



def failMessage(testName, actual, expected):
	print(Fore.RED + "[✗] : %s\nactual :\n%s\nexpected :\n%s"%(testName,str(actual),str(expected))+Style.RESET_ALL)

def passMessage(testName):
	print(Fore.GREEN + "[✔] : %s"%(testName)+Style.RESET_ALL)

def printThatMatrix(matrix, title=None):
	cmap = ListedColormap(["black","white"])
	imgplot = plt.imshow(matrix,cmap=cmap)
	if title!=None:
		plt.title(title)
		plt.show()
	else:
		plt.show()

def PrintMatricesInGrid(matrices):
	cmap = ListedColormap(["black","white"])
	
	rows = 3
	cols = len(matrices)//3
	if len(matrices)%10!=0:
		cols+=1

	print(len(matrices))
	
	fig, axes = plt.subplots(rows,cols)

	if isinstance(axes, np.ndarray):
		list_axes = list(axes.flat)
	else:
		list_axes = [axes]
	
	for i in range(len(matrices)):
		list_axes[i].imshow(matrices[i], cmap=cmap)
	
	fig.tight_layout()
	plt.show()

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

if __name__ == "__main__":
	test("equality", 1,1)
	test("equality2",1,2)
	test("equality3",1==1)
	test("equality4",np.array(np.linspace(0,9,10)),np.array(np.linspace(0,9,10)))
	test("equality5",np.array(np.linspace(0,9,10)),np.array(np.linspace(0,9,11)))