import numpy as np
import h5py
#from camel import Camel as CL
import camels_library as CL
import matplotlib as plt
import MAS_library as MASL

dataDM = h5py.File('/home/abhiram/PycharmProjects/MLxUniverse/Sims/IllustrisTNG_DM/snap_013.hdf5', 'r')
data = h5py.File('/home/abhiram/PycharmProjects/MLxUniverse/Sims/IllustrisTNG/snap_013.hdf5', 'r')

BoxSize = data['Header'].attrs[u'BoxSize']/1e3

print("Illustris", list(data.keys()))
print("Nbody ", list(dataDM.keys()))

dm = data['PartType1']
dm_DM = dataDM['PartType1']

pos_dm  = dm['Coordinates'][:]/1e3
vel_dm  = dm['Velocities'][:]*1e10

pos_dm_DM  = dm_DM['Coordinates'][:]/1e3
vel_dm_DM  = dm_DM['Velocities'][:]*1e10

print("size: ", (np.array(pos_dm)).size)

k = 32
threads = 1
verbose = True

radius_gas = CL.KDTree_distance(pos_dm,  pos_dm, k) #Mpc/h

grid_size = 256

map_dm = np.zeros((grid_size, grid_size), dtype=np.float64)
map_dm_DM = np.zeros((grid_size, grid_size), dtype=np.float64)

x_min, x_max = 0., BoxSize
y_min, y_max = 0., BoxSize
z_min, z_max = 0., 5.0

indexes = np.where(
    (pos_dm[:,0] >= x_min) & (pos_dm[:,0] < x_max) &
    (pos_dm[:,0] >= y_min) & (pos_dm[:,0] < y_max) &
    (pos_dm[:,0] >= z_min) & (pos_dm[:,0] < z_max)
)

indexes_DM = np.where(
    (pos_dm_DM[:,0] >= x_min) & (pos_dm_DM[:,0] < x_max) &
    (pos_dm_DM[:,0] >= y_min) & (pos_dm_DM[:,0] < y_max) &
    (pos_dm_DM[:,0] >= z_min) & (pos_dm_DM[:,0] < z_max)
)

pos_dm_     = pos_dm[indexes]
pos_dm_DM_  = pos_dm_DM[indexes]

pos_dm_ = pos_dm_[:,[0, 1]]
pos_dm_ = np.ascontiguousarray(pos_dm_)
pos_dm_ = pos_dm_.astype(np.float32)
print("pos_dm_ shape: ", pos_dm_.shape)

pos_dm_DM_ = pos_dm_DM_[:,[0, 1]]
pos_dm_DM_ = np.ascontiguousarray(pos_dm_)
pos_dm_DM_ = pos_dm_DM_.astype(np.float32)
print("pos_dm_ shape: ", pos_dm_DM_.shape)

area_pixel = (BoxSize/grid_size)**2



