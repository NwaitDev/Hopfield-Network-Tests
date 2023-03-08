import src.hopfieldnet as hn
from src.pltlib import printThatMatrix

if __name__ == "__main__":
   network = hn.trainAndDumpNetwork()
   printThatMatrix(network, "NETWORK")
