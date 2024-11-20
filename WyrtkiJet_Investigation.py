# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:16:53 2024

@author: Linne
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import pickle
from FuncsAPE import crop_oceanbasin, datapath
from EN4_singledepth import EN4_singledepth_time
import os
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import geopandas as gpd
from EOF import make_EOFsolver
from cartopy.util import add_cyclic_point
from scipy.signal import welch
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
sns.set_palette('plasma', n_colors = 4)


import matplotlib.path as mpath

log = False
#if density == True, take APE in J/kg, else take in J/m3
density = True
#setting time boundaries
startyear = 1960
endyear = 2023
spectral = True


#setting depth to choose
# depth = 400
#setting maximum (northmost) latitude for Southern Ocean. 
#Also the southern most latitude for SPO, SAO and IO
SO_cutoff_lat = -30
#setting number of eofs to plot
neofs = 1

with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
    
#reading in example data to make masks
datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
shape = data.salinity.squeeze().shape
mask = np.nan_to_num(surface_valid-1, nan = 1)

#getting lon and lat values for plotting
lon, lat = data.lon.to_numpy(), data.lat.to_numpy()
# lon[-1] = 0
LON, LAT = np.meshgrid(lon, lat)
fig, ax = plt.subplots(figsize = (7.3, 2.5))

for depth in [90, 100, 120, 130]:
# for depth in [100]:
    #reading in and taking data only at a single depth
    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], 0))
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    
    if log:
        log10 = np.log10(APE)
    
    
    OB = 'Indian Ocean'
    OF = ocean_filters[OB]
    masksum = np.isnan(OF) + mask
    mask_ocean = (masksum>0)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    
    
    if log:
        try:
            solver = make_EOFsolver(log10, mask_ocean)
        except:
            solver = 0
    else:
        try:
            solver = make_EOFsolver(APE, mask_ocean)       
        except:
            solver = 0
    #combining filters with land masks to make combined mask  
    
    if type(solver) != int:
        eof1 = solver.eofsAsCovariance(neofs=neofs)[0]
        pc1 = solver.pcs(npcs=neofs, pcscaling=1)
        fracs = solver.varianceFraction(neofs)
    #%%
    rolling_n = 12
    df = pd.DataFrame(pc1, columns = ['PC'])
    rolling = df.rolling(rolling_n, center = True)
    
    #%%
    oscillations = pc1[:, 0] - rolling.mean()['PC']
    mean = np.nanmean(oscillations.to_numpy().reshape(12, endyear - startyear + 1, order = 'F'), axis = 1)
    months = np.arange(1, 13)
    plt.plot(months, mean, label = str(int(np.round(true_depth, 0)))+'m')
    for i in range(len(months)):
        if len(str(months[i])) == 1:
            months[i] = '0'+ str(months[i])
        else:
            months[i] = str(months[i])
    print(eof1.max())
    
#%%
ax.set_xticks([1, 3, 5, 7, 9, 11], ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.axvline(5, label = 'Spring Jet Peak', linestyle = ':', color = 'black')
ax.axvline(11, label = 'Fall Jet Peak', linestyle = '--', color = 'black')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8,box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))
ax.legend(loc='upper left')

# ax.title(f'Indian Ocean ($\phi$>30S) mean of PC1 - {rolling_n} month rolling mean')
ax.set_ylabel('Detrended, Monthly Mean PC')
ax.set_xlabel('Month')
# plt.savefig('IO_100m_annualtrend_PC1.pdf', bbox_inches = 'tight')

plt.savefig('IO_multidepth_annualtrend_PC1.pdf', bbox_inches = 'tight')

