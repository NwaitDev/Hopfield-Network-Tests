import src.hopfieldnet as hn
from src.lib import *

if __name__ == "__main__":

   imgs = importImagesFromFolder("./img-data/8x8", 8, 8)

   network = hn.trainAndDumpNetwork(imgs,"network_8x8")
   print("Network shape : ", network.shape)
   printThatMatrix(network, "network_8x8")
