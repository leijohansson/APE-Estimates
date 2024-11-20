# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:14:12 2024

@author: Linne
"""

import pickle
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from FuncsAPE import datapath
import xarray as xr
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

sns.set_context('paper')

depth = 3000
if depth == 373:
    limits = [300, 310, -35, -45]
if depth == 3000:
    limits = [310, 320, -25, -40]


datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
temp = data.temperature.to_numpy().squeeze()[0, :, :]
depths = data.depth.to_numpy()

LON, LAT = np.meshgrid(data.lon, data.lat)

bool1 = (LON <= limits[1]).astype(int)
bool2 = (LON > limits[0]).astype(int)
bool3 = (LAT>limits[3]).astype(int)
bool4 = (LAT <=limits[2]).astype(int)

restrict = bool1+bool2+bool3+bool4
restrict = (restrict == 4).astype(float)
restrict[np.where(restrict==0)] = np.nan
# plt.imshow(restrict)
# %%
nmonths = (2024-1960)*12
x_time =  pd.date_range('1960-01-01', periods=nmonths, freq='m')


depth_diff = np.abs(depths - depth)
depth_i = np.where(depth_diff == np.min(depth_diff))[0]
depth_true = int(np.round(depths[depth_i], 0))
print(depth_true)


TS = np.zeros((nmonths))
i = 0

for year in range(1960, 2024):
    for month in range(1, 13):
        month = str(month)
        if len(month) == 1:
            month = '0'+month
        file = f'\APEarrays\APE_{year}-{month}.npy'
        APE = np.load(datapath + file)[depth_i, :, :]*restrict
        TS[i] = np.nanmean(APE)
        i += 1
    #%%
fig, ax= plt.subplots(figsize = (4, 3))

ax.plot(x_time, TS, zorder = 10, color = 'navy')
ax.spines[['right', 'top']].set_visible(False)
if depth_true == 373:
    ax2 = ax.inset_axes([0.45, 0.40, 0.5, 0.55])
    ax2.plot(x_time, TS, color = 'navy')
    ax2.set_xlim(x_time[215], x_time[235])
    ax2.xaxis.set_minor_locator(mdates.MonthLocator((1, 6)))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    
    
    ax.indicate_inset_zoom(ax2, color = 'gray', zorder = -5, edgecolor = 'black')
    

ax.set_ylabel('Mean APE density, J/kg')
ax.set_xlabel('Time')
# ax.set_title(r'Restricted Area: $-35^\circ <\phi \leq -45^\circ, -60^\circ<\lambda\leq -50^\circ, z = 373m$ ')
fig.savefig(f'weirdpeak{depth_true}.pdf', bbox_inches = 'tight')


#%%
OF = restrict
# valid_points = data.salinity.to_numpy().squeeze()/data.salinity.to_numpy().squeeze()*OF
temps = np.zeros((nmonths))*np.nan
sals = temps.copy()
timeidx = -1
for year in range(1960, 2024):
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
                #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
    
        filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
        data = xr.open_dataset(f'{datadir}/{filename}')
        temp_OW = data.temperature_observation_weights.to_numpy().squeeze()[depth_i, :, :].squeeze()
        sal_OW = data.salinity_observation_weights.to_numpy().squeeze()[depth_i, :, :].squeeze()

        #calculating APE up to depth 700
        # area_temp = A_3d * temp_OW * OF #temp takes care of valid area


        onemonth = np.nanmean(temp_OW*OF)#/depth_area
        temps[timeidx] = onemonth

        onemonthS = np.nanmean(sal_OW*OF)#/depth_area
        sals[timeidx] = onemonthS
#%%
fig, ax= plt.subplots(figsize = (6.3, 2))
ax.set_ylabel('Mean APE density, J/kg')
ax.set_xlabel('Time')
ax1 = ax.twinx()
ax1.plot(x_time, sals, label = 'Salinity', color = 'tab:pink', alpha = 0.6)
ax1.plot(x_time, temps, label = 'Temperature', color = 'tab:cyan', alpha = 0.6)
ax.plot(x_time, TS, color = 'black', zorder = 5)

ax1.set_ylabel('Mean Observation Weight')
if depth == 373:
    ax1.legend(loc = 'center right')
else:
    ax1.legend(loc = 'upper left')

fig.savefig(f'weirdpeak{depth_true}_OW.pdf', bbox_inches = 'tight')
