from contextlib import redirect_stderr
from mpi4py import MPI
from scipy import misc
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data, img, sz, original = None, None, None, None
if rank == 0:
    img = misc.imread("test3.PNG")
    if len(img.shape) > 2:
        img = img[:, :, 0]
    original = img

    img = np.reshape(img, (-1, 1))

    data = np.array_split(img, size)

sctared = comm.scatter(data, root=0)
freq = np.zeros((256))

for i in sctared:
    freq[i] += 1

reduced = comm.reduce(freq, op=MPI.SUM, root=0)

if rank == 0:
    data = np.array_split(reduced, size)
    sz = original.shape[0]*original.shape[1]

tot = comm.bcast(sz, root=0)
sctared = comm.scatter(data, root=0)
for i in range(len(sctared)):
    sctared[i] /= tot
    if i != 0:
        sctared[i] += sctared[i-1]

######################### NEW ###########################
help = 0

if rank != 0:
    help = comm.recv(source=(rank-1) % size)
if rank != size - 1:
    comm.send(sctared[-1] + help, dest=(rank+1) % size)


###################### END NEW #########################
for i in range(len(sctared)):
    sctared[i] = (sctared[i]+help)*255

freq = comm.gather(sctared, root=0)

if rank == 0:
    freq = np.concatenate(freq)
    data = np.array_split(img, size)

sctared = comm.scatter(data, root=0)
freq = comm.bcast(freq, root=0)


for i in range(len(sctared)):
    sctared[i] = int(freq[sctared[i]])

data = comm.gather(sctared, root=0)
if rank == 0:
    data = np.concatenate(data)
    data = np.reshape(data, (original.shape[0], original.shape[1]))

    misc.imshow(data)
    # misc.imsave('show6.png', original)
