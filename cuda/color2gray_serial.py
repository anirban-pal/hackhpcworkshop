import numpy as np
from matplotlib import pyplot as plt
from time import time

from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000 

# Read Images
infile = "/cm/shared/data/DIV8K_train_HR/1401.png"
img = plt.imread(infile)
orig = np.asarray(img)

time1 = time()    
gray = (0.2989 * orig[:,:,0] + 0.5870 * orig[:,:,1] + 0.1140 * orig[:,:,2])*255
time2 = time()

print("Time taken for matrix operation (sec nds):", time2-time1)

