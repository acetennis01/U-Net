import h5py
import numpy as np

import matplotlib.pyplot as plt

f = h5py.File('/home/abhiram/PycharmProjects/MLxUniverse/Sims/IllustrisTNG/snap_013.hdf5', 'r')
ls = list(f.keys())
print("List of the headers ", ls)
dset = f['PartType1']
dsetList = list(dset.keys())
dsetCoordinate = dset['Velocities']
array = np.array(dsetCoordinate)
array = array.reshape((256, 256, 256, 3))
firstArray = array[:, :, :, 1]

slices = np.empty(256, dtype=object)

for i in range(256):
    slices[i] = firstArray[:, :, i]

plt.imshow(slices[1], cmap='magma', interpolation='bicubic')
#plt.imshow(slices[1], cmap='magma')
plt.show()

print(firstArray.shape)
f.close()