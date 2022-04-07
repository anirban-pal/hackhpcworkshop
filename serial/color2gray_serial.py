import numpy as np
from matplotlib import pyplot as plt
from time import time

import os
try:
    os.mkdir("grayscale")
except OSError as error:
    pass

start_time = time()

num_imgs = 800
# Read Images
for i in range(1,num_imgs+1):
    
    infile = "/cm/shared/data/DIV2K_train_HR/{:04d}.png".format(i)
    outfile = "grayscale/{:04d}.png".format(i)
    
    img = plt.imread(infile)

    orig = np.asarray(img)
    gray = (0.2989 * orig[:,:,0] + 0.5870 * orig[:,:,1] + 0.1140 * orig[:,:,2])*255

    # Output Images
    plt.imsave(outfile, gray, cmap="gray")
    print(outfile)

finish_time = time()
elapsed_time = finish_time - start_time
print("Time taken (seconds):", elapsed_time)

