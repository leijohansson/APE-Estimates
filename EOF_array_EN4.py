# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:48:49 2024

@author: Linne
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
import scipy.io
import numpy.ma as ma
import pickle 
import pandas as pd
from EOF import make_EOFsolver
import matplotlib.gridspec as gridspec

startyear = 1960
endyear = 2022
neofs = 1
nyears = endyear + 1 - startyear
nmonths = nyears * 12
x_time =  pd.date_range(f'{startyear}-01-01', periods=nmonths, freq='m')

with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)


max_depth = np.inf

datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.temperature.squeeze().shape

depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]


#creating mask for surface (land vs ocean)
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)
   
depth_fracs = find_depthfracs(dz, shape, max_depth)

#calculating area represented by each gridpoint
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)

#creating 3d array of vertical distance represented by each grid point
#needed to find volume of each grid
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]
    

#setting up array for EOF (time, lat, lon)
timespace_arr = np.zeros((nmonths, len(data.lat), len(data.lon)))
timeidx = -1
for year in range(startyear, endyear+1):
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = f'APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(datapath + file)
        
        #calculating vertical integral of mean APE (all depths)
        vert_int = np.sum(APE_all, axis = 0)
        
        #masking non ocean areas
        VIm = ma.masked_array(vert_int, mask = mask)
        Am = ma.masked_array(A_ij, mask = mask)
        
        #calculating log10 of vertically integrated APE per unit area
        VI_per_area = vert_int/Am
        
        #code for log10 of vertically integrated
        # log_APEm2 = np.log10(VI_per_area).filled(np.nan)
        # #filling with 0 where value <0 (residual from negative APE values)
        # negative = np.where(log_APEm2<0)
        # log_APEm2[negative] = 0

        
        #vertically averaged
        zsum = np.sum(dz3*volume_valid, axis = 0)
        VA_per_area = VI_per_area / zsum
        timespace_arr[timeidx, :, :] = VA_per_area
        
        #finding log
        # log_APEm2 = np.log10(VA_per_area).filled(np.nan)
        #getting rid of negative values
        # negative_m2 = np.where(log_APEm2<0)
        # log_APEm2[negative] = 0
        
#EOF stuff here
#%%
# fig = plt.figure(figsize = (12, 12), constrained_layout = True)
# gs0 = gridspec.GridSpec(16, 4, figure=fig)

# ax1 = fig.add_subplot(gs0[0:2, :])
# ax2 = fig.add_subplot(gs0[2:4, :])
# ax3 = fig.add_subplot(gs0[4:8, 0:2])
# ax4 = fig.add_subplot(gs0[4:8, 2:])
# ax5 = fig.add_subplot(gs0[8:12, 0:2])
# ax6 = fig.add_subplot(gs0[8:12, 2:])
# ax7 = fig.add_subplot(gs0[12:, 1:3])

# ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

# fig1 = plt.figure()
# gs1 = gridspec.GridSpec(4, 4, figure=fig1)

# ax11 = fig1.add_subplot(gs0[0, 0:2])
# ax12 = fig1.add_subplot(gs0[0, 2:4])
# ax13 = fig1.add_subplot(gs0[1, 0:2])
# ax14 = fig1.add_subplot(gs0[1, 2:4])
# ax15 = fig1.add_subplot(gs0[2, 0:2])
# ax16 = fig1.add_subplot(gs0[2, 2:4])
# ax17 = fig1.add_subplot(gs0[3, 1:3])

# ax1_list = [ax11, ax12, ax13, ax14, ax15, ax16, ax17]


fig, axs = plt.subplots(4, 2, figsize=(12, 15))
fig1, axs1 = plt.subplots(4, 2, figsize=(12, 15))

oceankeys = ['Arctic Ocean', 'Southern Ocean', 'North Atlantic Ocean', 
             'North Pacific Ocean', 'South Atlantic Ocean', 
             'South Pacific Ocean', 'Indian Ocean']
f_i = 0
lon, lat = data.lon.to_numpy(), data.lat.to_numpy()
for OB in oceankeys:
    OF = ocean_filters[OB]
    masksum = np.isnan(OF) + mask
    mask_ocean = (masksum>0)
    solver = make_EOFsolver(timespace_arr, mask_ocean)
    
    eof1 = solver.eofsAsCovariance(neofs=neofs)
    pc1 = solver.pcs(npcs=1, pcscaling=1)
    fracs = solver.varianceFraction(neofs)

    basin, lonb, latb = crop_oceanbasin(eof1[0, :, :], lon, lat)
    extent = [lonb[0], lonb[-1], latb[0], latb[-1]]
    # ax = ax_list[f_i]
    ax = axs[f_i//2, f_i%2]
    ax.set_facecolor('darkgrey')
    plot = ax.imshow(np.flip(basin, axis = 0), #levels=clevs,
                cmap=plt.cm.RdBu_r, extent = extent)
    fig.colorbar(plot, ax = ax, shrink=0.6)
    ax.set_title(f'{OB} EOF1: {round(fracs[0], 4)*100}%')

    ax1 = axs1[f_i//2, f_i%2]
    # ax1 = ax1_list[f_i]
    ax1.plot(x_time, pc1)
    ax1.set_title(f'{OB} EOF1: {round(fracs[0], 4)*100}%')

    f_i += 1
    
fig.tight_layout()
fig1.tight_layout()

fig.savefig('EN4 Plots/Density_EOF_spatial.pdf', bbox_inches = 'tight')
fig1.savefig('EN4 Plots/Density_EOF_PC1.pdf', bbox_inches = 'tight')

    
