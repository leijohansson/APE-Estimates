# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:57:03 2024

@author: Linne

Comparison of APE values to those published by Huang 2005
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
import scipy.io
import numpy.ma as ma
import pickle 
import pandas as pd

#grid spacing 0.25 deg
datadir = datapath + 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

#creating mask for surface (land vs ocean)
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
mask = np.nan_to_num(surface_valid-1, nan = 1)

#creating 3d bool array for ocean (1) and no ocean (0)
volume_s = data.salinity.squeeze().to_numpy()
volume_valid = np.nan_to_num(volume_s/volume_s, nan = 0)

#finding vertical distance represented by each grid point
depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i

#calculating area represented by each gridpoint
A_ij = calc_Aij(data)
A_ij3 = np.broadcast_to(A_ij, shape)

#creating 3d array of vertical distance represented by each grid point
dz3 = np.zeros(shape)
for i in range(len(dz)):
    dz3[i, :, :] = dz[i]

# finding depth fracs for depth up to 5750m
depth_fracs = find_depthfracs(dz, shape, 5750)
#volume top to bottom
V_ijk = dz3*A_ij3
#restricting volume to set depth
V_ijk *= depth_fracs
#accounting for topography in volume
V_ijk *= volume_valid

#open ocean and sea filters
with open('RegionFilters/ocean_filters-WOCE_-45.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)

#creating filters for atlantic,  pacific, indian, and world oceans
atlantic = -(np.isnan(ocean_filters['North Atlantic Ocean'])-1) -\
    (np.isnan(ocean_filters['South Atlantic Ocean'])-1) -\
        (np.isnan(ocean_filters['Arctic Ocean'])-1)
    
pacific = -(np.isnan(ocean_filters['North Pacific Ocean'])-1) -\
    (np.isnan(ocean_filters['South Pacific Ocean'])-1)
    
indian = -(np.isnan(ocean_filters['Indian Ocean'])-1)

medi = -(np.isnan(ocean_filters['Mediterranean Region'])-1) 
southern = -(np.isnan(ocean_filters['Southern Ocean'])-1) 


world_oceans = np.zeros(shape[1:])
for OB in ocean_filters.keys():
    if 'Ocean' in OB:
        print(OB)
        world_oceans -= (np.isnan(ocean_filters[OB])-1)
#%%
#creating filters for atlantic + medi and world oceans + medi
atlantic_medi = atlantic + medi

world_oceans_medi = world_oceans + medi

labels = ['Atlantic with Medi', 'Atlantic', 'Pacific', 'Indian', 'World Oceans with Medi', 'World Oceans', 'Southern']
filters = [atlantic_medi, atlantic, pacific, indian, world_oceans_medi, world_oceans, southern]

#calculating APE density for each filter
df = pd.DataFrame(columns = ['BAR', 'PYC'], index = labels)
for method in ['BAR', 'PYC']:
    filename = datapath + f'WOCE_Data/APEarrays/WAGHC_APE_{method}-mean.npy'
    mean_APE = np.load(filename)
    print(f'method: {method}')
    densities = np.zeros(len(filters))
    for i in range(len(filters)):
        basin = filters[i]
        label = labels[i]
        basin_APE = np.sum(basin*mean_APE)
        volume = np.sum(basin*V_ijk)
        density = basin_APE/volume
        densities[i] = density
    df[method] = densities
#saving df with calculated values and Huang values
df['Huang'] = [708.0, 638.8, 481.7, 472.7, 664.4, 624.2, np.nan]
# df.to_csv('LiteratureComparisons/APE_densities_Huang.csv')

#creating list of months for plotting
months = np.arange(1,13).astype(str)
for i in range(len(months)):
    if len(months[i]) == 1:
        months[i] = '0'+months[i]
#%%
#Plotting APE density against month for different filters
fig, axs = plt.subplots(4, 2, sharex=True, figsize = (15, 12))
ax = 0
for method in ['BAR', 'PYC']:
    print(f'method: {method}')
    df = pd.DataFrame(columns = months, index = labels)
    for month in months:
        filename = datapath + f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        mean_APE = np.load(filename)
        densities = np.zeros(len(filters))
        for i in range(len(filters)):
            basin = filters[i]
            label = labels[i]
            basin_APE = np.sum(basin*mean_APE)
            volume = np.sum(basin*V_ijk)
            density = basin_APE/volume
            densities[i] = density
        df[month] = densities
    
    for i in range(len(df)):
        axs[i//2, i%2].plot(months, df.iloc[i], label = method)
        axs[i//2, i%2].set_title(df.index[i])
        axs[i//2, i%2].set_ylabel('APE density, $Jm^{-3}$')

axs[3, 0].set_xlabel('Month')
axs[3, 1].set_xlabel('Month')

axs[0, 1].legend()
fig.savefig('WOCE Plots/Density_Climatology_WOCE_huang.png', bbox_inches = 'tight')
# plt.close()
