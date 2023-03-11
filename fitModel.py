import src.hopfieldnet as hn
from lib import printThatMatrix

if __name__ == "__main__":
   network = hn.trainAndDumpNetwork()
   printThatMatrix(network, "NETWORK")
