# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:26:26 2024

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
sns.set_context('paper')


SO_cutoff_lat = -45

with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
ocean_filters['World'] = np.ones(ocean_filters['Indian Ocean'].shape)

with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    IO30 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
IO_band = IO45-IO30
ocean_filters['IO Band'] = IO_band

datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
temp = data.temperature.to_numpy().squeeze()[0, :, :]

valid = temp/temp

oceans = ['North Atlantic Ocean', 'South Atlantic Ocean',
          'North Pacific Ocean', 'South Pacific Ocean',
          'Indian Ocean', 'Southern Ocean', 'Arctic Ocean',
          'Mediterranean Region', 'Baltic Sea',
          'South China and Easter Archipelagic Seas']

plot_arr = np.zeros(ocean_filters['World'].shape)

i = 0
for OB in oceans:
    i += 1
    plot_arr += ~np.isnan(ocean_filters[OB])*i*valid
    
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize = (6.3, 4))
plot_arr[np.where(plot_arr == 0)] = np.nan
basin_c, lon_c = add_cyclic_point(plot_arr, coord=data.lon, axis=-1)

# plot = ax.contourf(lon_c, data.lat, basin_c, cmap = "Set3", levels = 10)
plot = ax.imshow(np.flip(basin_c, axis = 0), vmin = 0.5, vmax = 12.5,
                 extent = [lon_c[0], lon_c[-1], data.lat[0], data.lat[-1]], cmap = "Set3")

ax.coastlines()
gl = ax.gridlines(draw_labels=True, alpha = 0)
gl.xlabels_top = False
gl.ylabels_right= False
# plt.colorbar(plot)

fs = 7
plt.text(-65, 30, f'''North Atlantic''', fontsize = fs)
plt.text(-40, -25, f'''South Atlantic''', fontsize = fs)
plt.text(-170, 30, f'''North Pacific''', fontsize = fs)
plt.text(-150, -25, f'''South Pacific''', fontsize = fs)
plt.text(65, -25, f'''Indian''', fontsize = fs)
plt.text(-10, 75, f'''Arctic''', fontsize = fs)
plt.text(-20, -60, f'''Southern''', fontsize = fs)
plt.text(10, 15, f'''Medi''', fontsize = fs)
plt.arrow(20, 59, 10, -8, head_width = 3, length_includes_head = 1, fc = 'black')
plt.text(30, 50, f'''Baltic''', fontsize = fs)
plt.arrow(18, 37, 0, -10, head_width = 3, length_includes_head = 1, fc = 'black')
plt.text(135, 15, f'''South China''', fontsize = fs)
plt.arrow(112, 8, 22, 9, head_width = 3, length_includes_head = 1, fc = 'black')

plt.savefig('seas.pdf')
