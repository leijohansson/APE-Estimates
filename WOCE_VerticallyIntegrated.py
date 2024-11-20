# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:30:45 2024

@author: Linne
"""
import numpy as np 
import matplotlib.pyplot as plt
from FuncsAPE import datapath, calc_Aij
import xarray as xr
import cartopy.crs as ccrs


method = 'PYC'
month = '01'

datadir = datapath + 'WOCE_Data/Data/'
filename = f'WAGHC_BAR_01_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.temperature.squeeze().shape
lon = data.longitude.to_numpy()
lat = data.latitude.to_numpy()
surface_s = data.salinity.to_numpy().squeeze()[0, :, :]
surface = surface_s/surface_s
extent = [lon[0], lon[-1], lat[0], lat[-1]]

APE_months = np.ones((12, shape[1], shape[2]))

i = 0
for month in range(1, 13):
    if len(str(month)) == 1:
        #eg '1' becomes '01' (as in the filenames)
        month = '0'+str(month)

    file = f'{datapath}/WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
    APE = np.load(file) #APE in J
    APE_months[i,:, :] = np.sum(APE, axis = 0)

APE_months[np.where(APE_months == 0)] = np.nan
APE_months[np.where(APE_months== 1)] = np.nan
mean_APE_VI = np.nanmean(APE_months, axis = 0)/calc_Aij(data)

#%%
# for i in range(12):
# plt.contourf(lon, lat, np.log10(mean_APE_VI*surface), levels = 20, cmap = 'bwr')
    # plt.figure()
    # plot = plt.contourf(lon, lat, np.log10(APE_months[0, :, :]*surface), levels = 20, cmap = 'coolwarm', vmin = 8)
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

plot = ax.imshow(np.flip(np.log10(mean_APE_VI*surface), axis = 0), cmap = 'coolwarm', vmin =  4, extent = extent)
plt.colorbar(plot, location = 'bottom', label = 'Annual Mean $log_{10}$(Vertically Integrated APE []) ')
ax.coastlines()
