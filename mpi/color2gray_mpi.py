import numpy as np
from matplotlib import pyplot as plt
from time import time
from mpi4py import MPI

import os
try:
    os.mkdir("grayscale")
except OSError as error:
    pass

start_time = time()

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()

num_imgs = 800
for i in range(1,num_imgs+1):
    
	if (i % world_size == my_rank):
		
		# Read Images
		infile = "/cm/shared/data/DIV2K_train_HR/{:04d}.png".format(i)
		img = plt.imread(infile)
		
		# Convert image to grayscale
		orig = np.asarray(img)
		gray = (0.2989 * orig[:,:,0] + 0.5870 * orig[:,:,1] + 0.1140 * orig[:,:,2])*255
		
		# Output Images
		outfile = "grayscale/{:04d}.png".format(i)
		plt.imsave(outfile, gray, cmap="gray")
		print(outfile, "processor ", my_rank)

world_comm.Barrier()

if (my_rank == 0):
	finish_time = time()
	elapsed_time = finish_time - start_time
	print("Time taken (seconds):", elapsed_time, " with ",world_size, " processors." )

