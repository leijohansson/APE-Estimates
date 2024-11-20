# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:08:00 2024

@author: Linne
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from cartopy.util import add_cyclic_point
from FuncsAPE import datapath, calc_Aij
from matplotlib.colors import TwoSlopeNorm

#%%
countries = gpd.read_file(datapath+'ne_110m_land.zip')


datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
shape = data.salinity.squeeze().shape
#%%

startyear1 = 1960
endyear1 = 2000
nmonths1 = (endyear1 - startyear1 +1)*12

APE1 = np.zeros((nmonths1, len(data.lat), len(data.lon)))

timeidx = -1
for year in range(startyear1, endyear1+1):
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = f'{datapath}APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        onemonth = np.sum(APE_all, axis = 0)
        APE1[timeidx, :, :] = onemonth
APE1 = np.mean(APE1, axis = 0)

startyear2 = 2003
endyear2 = 2023
nmonths2 = (endyear2 - startyear2 +1)*12

APE2 = np.zeros((nmonths2, len(data.lat), len(data.lon)))
timeidx = -1
for year in range(startyear2, endyear2+1):
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = f'{datapath}APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        onemonth = np.sum(APE_all, axis = 0)
        APE2[timeidx, :, :] = onemonth
APE2 = np.mean(APE2, axis = 0)

        
diff = APE2 - APE1

with open('RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
OF = ocean_filters['Indian Ocean']

diff = (APE2 - APE1)*OF
#%%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (4, 4))
diff_c, lon_c = add_cyclic_point(diff, coord=data.lon, axis=-1)

plot = ax.contourf(lon_c, data.lat, diff_c, cmap = 'viridis')
plt.colorbar(plot, location = 'bottom', label = "$\Delta APE, Jm^{-2}$")

ax.coastlines()
gls = ax.gridlines(draw_labels = True, color = 'black', alpha = 0.2)
gls.top_labels=False
gls.right_labels=False 

lat, lon = data.lat.to_numpy(), data.lon.to_numpy()
lon_where_1d = np.sum(OF>0, axis = 0)
lonidx = np.where(OF >0)[0]
lat_where_1d = np.sum(OF>0, axis = 1)
latidx = np.where(lat_where_1d >0)[0]
latlims = [lat[latidx[0]], lat[latidx[-1]]]

if True in ((lonidx[1:] - lonidx[0:-1]) > 100):
    idx_gap_1 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]]
    idx_gap_2 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]+1]

    lonlims = [lon[idx_gap_2], lon[idx_gap_1]]

else:
    lonlims = [lon[lonidx[0]], lon[lonidx[-1]]]
ax.set_xlim(lonlims[0], lonlims[-1])
ax.set_ylim(latlims[0], latlims[-1])
plt.savefig('IO_meandiff.pdf', bbox_inches = 'tight')
#%%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
fig.suptitle('\overline{APE}_{2003-2023})$- $log_{10}(\overline{APE}_{1960-1990}')
basin_c, lon_c = add_cyclic_point(np.log10(diff), coord=data.lon, axis=-1)
basin_c_neg, lon_c = add_cyclic_point(np.log10(-diff), coord=data.lon, axis=-1)

plot = ax.contourf(lon_c, data.lat, basin_c, cmap = 'Blues')
plt.colorbar(plot, location = 'bottom', label = "log10 positive APE'")

plot = ax.contourf(lon_c, data.lat, basin_c_neg, cmap = 'Reds')
plt.colorbar(plot, location = 'bottom', label = "log10 negative APE'")

ax.coastlines()

lat, lon = data.lat.to_numpy(), data.lon.to_numpy()
lon_where_1d = np.sum(OF>0, axis = 0)
lonidx = np.where(OF >0)[0]
lat_where_1d = np.sum(OF>0, axis = 1)
latidx = np.where(lat_where_1d >0)[0]
latlims = [lat[latidx[0]], lat[latidx[-1]]]

if True in ((lonidx[1:] - lonidx[0:-1]) > 100):
    idx_gap_1 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]]
    idx_gap_2 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]+1]

    lonlims = [lon[idx_gap_2], lon[idx_gap_1]]

else:
    lonlims = [lon[lonidx[0]], lon[lonidx[-1]]]
extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
ax.set_extent(extent)

#%%
fig1, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
fig1.suptitle('Indian Ocean Depth Integrated APE')
basin_2, lon_c = add_cyclic_point(np.log10(APE2), coord=data.lon, axis=-1)
plot = axs[0].contourf(lon_c, data.lat, basin_2, cmap = 'jet')
plt.colorbar(plot, location = 'bottom', label = '$log_{10}(\overline{APE})$', ax = axs[0])
axs[0].coastlines()
axs[0].set_title('2003-2023')
axs[0].set_extent(extent)

basin_1, lon_c = add_cyclic_point(np.log10(APE1), coord=data.lon, axis=-1)
plot = axs[1].contourf(lon_c, data.lat, basin_1, cmap = 'jet')
plt.colorbar(plot, location = 'bottom', label = '$log_{10}(\overline{APE})$', ax = axs[1])
axs[1].coastlines()
axs[1].set_title('1960-2000')
axs[1].set_extent(extent)

#%%
