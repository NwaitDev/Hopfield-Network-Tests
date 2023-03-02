import numpy as np
from colorama import Fore, Style

def failMessage(testName, actual, expected):
	print(Fore.RED + "[✗] : %s\nactual :\n%s\nexpected :\n%s"%(testName,str(actual),str(expected)))
	print(Style.RESET_ALL)

def passMessage(testName):
	print(Fore.GREEN + "[✔] : %s"%(testName))
	print(Style.RESET_ALL)

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