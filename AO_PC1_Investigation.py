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
import matplotlib.dates as mdates
from EN4_EOF_onedepth import polarCentral_set_latlim
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
    
depths = [200, 400, 600, 800]
fig, axs = plt.subplots(len(depths)//2, 2, figsize=(6, 5),
                        subplot_kw={'projection': ccrs.NorthPolarStereo()}, 
                        layout = 'constrained')
fig2, axs2 = plt.subplots(len(depths)//2, 2, figsize=(6, 3), 
                          layout = 'constrained', sharex = True)

fig1, ax1 = plt.subplots(figsize = (6.3, 2.5))
for i in range(len(depths)):
    depth = depths[i]
    plotdir = f'EN4 Plots/EOF_{depth}m'
    if f'EOF_{depth}m' not in os.listdir(path = 'EN4 Plots/'):
        os.mkdir(f'EN4 Plots/EOF_{depth}m')
    
    #reading in and taking data only at a single depth
    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], 0))
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    
    if log:
        log10 = np.log10(APE)
    
    
    OB = 'Arctic Ocean'
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
    if depth == 200:
        eof1 = -eof1
        pc1 = -pc1
    axs[i//2, i%2].coastlines()
    basin, lonb, latb = crop_oceanbasin(eof1, lon, lat)

    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)

    basin_c, lon_c = add_cyclic_point(basin, coord=lon, axis=-1)
    plot = axs[i//2, i%2].contourf(lon_c, latb, basin_c, #levels=clevs,
                cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
    lat_lims = [latb[0], latb[-1]]
    polarCentral_set_latlim(lat_lims, axs[i//2, i%2])
    if density:
        label = 'APE Variance, $Jkg^{-1}$'
    else:
        label = 'APE Variance, $Jm^{-3}$'
    cbar = fig.colorbar(plot, ax = axs[i//2, i%2], shrink=0.75, location = 'right', label = label)
    for label in cbar.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    axs[i//2, i%2].set_title(f'{true_depth}m')
    axs[i//2, i%2].set_facecolor('darkgray')
    #%%
    rolling_n = 12
    df = pd.DataFrame(pc1, columns = ['PC'])
    rolling = df.rolling(rolling_n, center = True)
    
    axs2[i//2, i%2].plot(time, pc1, label = 'EOF', color = 'royalblue')
    axs2[i//2, i%2].plot(time, rolling.mean()['PC'], color = 'black',
                         label = f'{rolling_n} month rolling mean')
    axs2[i//2, i%2].spines[['right', 'top']].set_visible(False)

    axs2[1, i%2].set_xlabel('Year')
    axs2[i//2, i%2].set_ylabel('PC1')
    axs2[i//2, i%2].set_title(f'{true_depth}m')
    # axs2[i//2, i%2].xaxis.set_major_locator(mdates.YearLocator(20))

    # gl = axs[i//2, i%2].gridlines(draw_labels=True, xlocs=None, ylocs=None, color = 'black', alpha = 0.3)
    # gl.n_steps = 6

    

    #%%
    oscillations = pc1[:, 0] - rolling.mean()['PC']
    mean = np.nanmean(oscillations.to_numpy().reshape(12, endyear - startyear + 1, order = 'F'), axis = 1)
    months = np.arange(1, 13)
    for i in range(len(months)):
        if len(str(months[i])) == 1:
            months[i] = '0'+ str(months[i])
        else:
            months[i] = str(months[i])
    
    ax1.plot(months, mean, label = f'{true_depth}m')
ax1.set_xticks([1, 3, 5, 7, 9, 11], ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov'])
ax1.xaxis.set_minor_locator(MultipleLocator(2))
ax1.legend()
# ax.set_title(f'Southern Ocean mean of PC1 - {rolling_n} month rolling mean')
ax1.set_ylabel('Detrended, Monthly Mean PC')
ax1.set_xlabel('Month')
ax1.spines[['right', 'top']].set_visible(False)



# fig.constrained_layout()
# fig1.tight_layout('constrained')
# fig2.tight_layout()
fig.savefig('AO_4depths_EOFs_spatial.pdf', bbox_inches = 'tight')
fig2.savefig('AO_4depths_EOFs_PCs.pdf', bbox_inches = 'tight')
fig1.savefig('AO_4depths_EOFs_annual.pdf', bbox_inches = 'tight')
