import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
from time import time

from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 

# Read Images
infile = "/cm/shared/data/DIV8K_train_HR/1401.png"
img = plt.imread(infile)

orig = cp.asarray(img)

time1 = time()    
gray = (0.2989 * orig[:,:,0] + 0.5870 * orig[:,:,1] + 0.1140 * orig[:,:,2])*255
time2 = time()

print("Time taken for matrix operation (seconds):", time2-time1)

