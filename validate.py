from src.testThatShit import test
import numpy as np
import hopfieldnet as hn
import src.imageToMatrix as itm


if __name__ == "__main__":
	img = [[0,1],[0,1]]
	expected = np.array([[0,0,0,0],[0,0,0,1],[0,0,0,0],[0,1,0,0]])
	actual = hn.networkFromImages([img],imgHeight=2,imgWidth=2)
	
	test("test testing",actual,expected)
