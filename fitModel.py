import src.hopfieldnet as hn
from src.lib import *

if __name__ == "__main__":

   imgs = importImagesFromFolder("./img-data/simpsons", 64,64)
   network = hn.trainAndDumpNetwork(imgs,"network_simpsons_allimgs")
   PrintMatricesInGrid(imgs)
   printThatMatrix(network, "network_trained")
