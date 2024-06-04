# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:12:51 2024

@author: Linne
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
#grid spacing 0.25 deg
import scipy.io
import numpy.ma as ma
import pickle 

#whether to plot and save vertically integrated APE for each month 
plot = False

#maximum depth
depth_co = 700

#read data
datadir = 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

#get reference pressure
z1D = data.depth.to_numpy()
z = np.zeros(shape)
for i in range(len(z1D)):
    z[i, :, :] = z1D[i]
p = pr(z) 

#calculate vertical distance between grid points
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = (depths[1]-depths[0])/2

#layer method
# depth_sum = dz[0]
# for i in range(1, len(depths)):
#     dz_i = (depths[i]- depth_sum)*2
#     dz[i] = dz_i
#     depth_sum += dz_i

#point method
for i in range(1, len(depths)-1):
    dz[i] = 0.5*(depths[i+1] - depths[i-1])
dz[-1] = 0.5*(depths[-1] - depths[-2])


# creating 2D array of area covered by each grid point
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]

# creating 3D array of volume covered by each grid point
V_ijk = dz3*A_ij3

#creating mask for topography
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#finding fraction of each depth needed for the maximum depth
# 3d array
depth_fracs = find_depthfracs(dz, shape)

#for each data type
for method in ['PYC', 'BAR']:
    APEdict = {}
    for month in range(1, 13):
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
             month = '0'+str(month)
        filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
        #calculate BGE, APE
        BGE, APE_dV = calc_APE(datadir, filename, V_ijk, p, z)
        
        #calculate vertically integrated APE up to depth of 700m
        APE_700 = np.sum(APE_dV*depth_fracs, axis = 0)
        
        #mask topography
        mAPE = ma.masked_array(APE_700, mask = mask)
        if plot:
            fig, ax = plt.subplots()
            ax.set_facecolor('lightgrey')
            extent = [data.longitude[0], data.longitude[-1], data.latitude[0], data.latitude[-1]]
            plot = ax.imshow(np.flip(mAPE, axis = 0), extent = extent)
            ax.set_title(f'WOCE APE 700m {method}-{month}: {np.round(np.sum(APE_700), -15)}J')
            plt.colorbar(plot, location = 'bottom')
            ax.set_ylabel('Latitude, $^\circ$')
            ax.set_xlabel('Longitude, $^\circ$')
            fig.savefig(f'WOCE Plots/{method}/Absolute/{method}-{month}-700m.png', 
                        bbox_inches = 'tight')
            plt.close()
        
        #save 3D APE array
        np.save(f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy', APE_dV)
        #saving volume integrated depth up to 700m for each month
        APEdict[str(month)] = np.sum(APE_700)
        
    #save monthly 700m APE dictionary
    with open(f'{method}_APE700.pkl', 'wb') as f:
        pickle.dump(APEdict, f)
        




