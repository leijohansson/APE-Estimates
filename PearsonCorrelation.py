# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:03:46 2024

@author: Linne
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from eofs.standard import Eof
import cartopy.crs as ccrs
import pickle
from FuncsAPE import crop_oceanbasin, datapath
from scipy.stats import pearsonr
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm
from EN4_EOF_onedepth import polarCentral_set_latlim


with open('RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    IO30 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
IO_band = IO45-IO30
IO_band[np.where(IO_band == 0)] = np.nan
ocean_filters['IO Band'] = IO_band

startyear = 1960
endyear = 2023
nyears = endyear + 1 - startyear
nmonths = nyears * 12
x_time =  pd.date_range(f'{startyear}-01-01', periods=nmonths, freq='m')

neofs = 1

datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
lat = data.lat.to_numpy()
lon = data.lon.to_numpy()
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
shape = data.salinity.squeeze().shape
mask = np.nan_to_num(surface_valid-1, nan = 1)

norm = TwoSlopeNorm(vcenter=0, vmin = -1, vmax = 1)

#setting up array for EOF (time, lat, lon)
timespace_arr = np.zeros((nmonths, len(data.lat), len(data.lon)))


timeidx = -1
for year in range(startyear, endyear+1):
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = f'{datapath}APEarrays\APE_{year}-{month}.npy'
        APE_all = np.load(file)
        APE_700 = np.sum(APE_all, axis = 0)
        mAPE = ma.masked_array(APE_700, mask = mask)
        timespace_arr[timeidx, :, :] = mAPE
#%%

for OB in ocean_filters.keys():
    basin_data = timespace_arr*ocean_filters[OB]
    corr_arr = np.zeros(surface_s.shape)*np.nan
    TS = np.nansum(basin_data, axis = (1,2))
    
    lon_where_1d = np.sum(basin_data[0, :, :]<np.inf, axis = 0)
    lonidx = np.where(lon_where_1d >0)[0]
    lat_where_1d = np.sum(basin_data[0, :, :]<np.inf, axis = 1)
    latidx = np.where(lat_where_1d >0)[0]
    latlims = [lat[latidx[0]], lat[latidx[-1]]]
    
    if True in ((lonidx[1:] - lonidx[0:-1]) > 100):
        idx_gap_1 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]]
        idx_gap_2 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]+1]
        lonlims = [lon[idx_gap_2], lon[idx_gap_1]]

    else:
        lonlims = [lon[lonidx[0]], lon[lonidx[-1]]]
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
        
    if OB == 'Arctic Ocean':
        fig, ax = plt.subplots(1, 1 , figsize=(5, 5),
                                subplot_kw={'projection': ccrs.NorthPolarStereo()})
    elif OB == 'Southern Ocean':
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                                subplot_kw={'projection': ccrs.SouthPolarStereo()})
    elif 'Pacific' in OB:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude = 180)})

    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                                subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    ax.set_facecolor('grey')
    
    for i in range(basin_data.shape[1]):
        for j in range(basin_data.shape[2]):
            if not np.isnan(basin_data[0, i, j]):
                corr_arr[i, j] = pearsonr(TS, basin_data[:, i, j]).statistic
    
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        corr_arr, lonb, latb = crop_oceanbasin(corr_arr, lon, lat)
        
        ax.coastlines()
        basin_c, lon_c = add_cyclic_point(corr_arr, coord=lon, axis=-1)
        plot = ax.contourf(lon_c, latb, basin_c, #levels=clevs,
                    cmap=plt.cm.RdBu_r, norm = norm, transform = ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, xlocs=None, ylocs=None, color = 'black', alpha = 0.3)
        gl.n_steps = 90
        lat_lims = [latb[0], latb[-1]]
        polarCentral_set_latlim(lat_lims, ax)
    else:
        ax.coastlines()
        ax.gridlines(draw_labels = True, color = 'black', alpha = 0.3)

        ax.set_extent(extent)
        basin_c, lon_c = add_cyclic_point(corr_arr, coord=lon, axis=-1)
        plot = ax.contourf(lon_c, lat, basin_c, #levels=clevs,
                    cmap=plt.cm.RdBu_r, norm = norm, transform = ccrs.PlateCarree())
    plt.colorbar(plot, location = 'bottom')
    ax.set_title(f'Pearson Correlation With {OB} volume integrated TS, {startyear}-{endyear}')
    plt.savefig(f'EN4 Plots/PearsonCorrelation/{OB}_Correlation.png', bbox_inches = 'tight')
    
    
                