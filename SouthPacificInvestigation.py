# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:18:18 2024

@author: Linne
"""


import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import pickle
from FuncsAPE import crop_oceanbasin, datapath, calc_Aij
from EN4_singledepth import EN4_singledepth_time
import os
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import geopandas as gpd
from EOF import make_EOFsolver
from cartopy.util import add_cyclic_point
from scipy.signal import welch
from EOF_final import EOF_ocean, find_lonlatlims, find_proj
import matplotlib.dates as mdates

import matplotlib.path as mpath
countries = gpd.read_file(datapath+'ne_110m_land.zip')

#from https://nordicesmhub.github.io/NEGI-Abisko-2019/training/example_NorthPolarStereo_projection.html

#reading in example data to make masks
datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
shape = data.salinity.squeeze().shape
mask = np.nan_to_num(surface_valid-1, nan = 1)
A_ij = calc_Aij(data)

lat = data.lat
lon = data.lon
#whether to use depth integrated APE
integrated = True
#setting depth to choose
depth = 800

#setting time boundaries
startyear = 1960
endyear = 2023
nmonths = (endyear + 1 - startyear) * 12

fs = 12


#setting number of eofs to plot
neofs = 4

plotdir = f'EN4 Plots/EOF_{depth}m'
if f'EOF_{depth}m' not in os.listdir(path = 'EN4 Plots/'):
    os.mkdir(f'EN4 Plots/EOF_{depth}m')

#reading in and taking data only at a single depth
if integrated:
    APE = np.zeros((nmonths, len(data.lat), len(data.lon)))
    timeidx = -1
    for year in range(startyear, endyear+1):
        for month in range(1, 13):
            timeidx += 1
            if len(str(month)) == 1:
                #eg '1' becomes '01' (as in the filenames)
                month = '0'+str(month)
    
            file = f'{datapath}APEarrays\APE_{year}-{month}.npy'
            APE_all = np.load(file)
            #calculating APE up to depth 700
            onemonth = np.sum(APE_all, axis = 0)/A_ij
            APE[timeidx, :, :] = onemonth
    time =  pd.date_range(f'{startyear}-01-01', periods=nmonths, freq='m')
            
else:
    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))


#reading in filters for different ocean basins
with open('RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
    

#getting lon and lat values for plotting
lon, lat = data.lon.to_numpy(), data.lat.to_numpy()
extent = [lon[0], lon[-1], lat[0], lat[-1]]

# lon[-1] = 0
LON, LAT = np.meshgrid(lon, lat)
#%%
OB = 'South Pacific Ocean'
n = 0
cbarloc = 'bottom'
OF = ocean_filters[OB]
masksum = np.isnan(OF) + mask
mask_ocean = (masksum>0)

lon_m = ma.masked_array(LON, mask = mask_ocean)
lat_m = ma.masked_array(LAT, mask = mask_ocean)



try:
    solver = make_EOFsolver(APE, mask_ocean)       
except:
    solver = 0
#combining filters with land masks to make combined mask

    

if type(solver) != int:
    eof1 = solver.eofsAsCovariance(neofs=neofs)
    pc1 = solver.pcs(npcs=neofs, pcscaling=1)
    fracs = solver.varianceFraction(neofs)
    # basin, lonb, latb = crop_oceanbasin(eof1[0, :, :], lon, lat)
    
    
    lon_where_1d = np.sum(eof1[0, :, :]<np.inf, axis = 0)
    lonidx = np.where(lon_where_1d >0)[0]
    lat_where_1d = np.sum(eof1[0, :, :]<np.inf, axis = 1)
    latidx = np.where(lat_where_1d >0)[0]
    latlims = [lat[latidx[0]], lat[latidx[-1]]]

    if True in ((lonidx[1:] - lonidx[0:-1]) > 100):
        idx_gap_1 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]]
        idx_gap_2 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]+1]

        lonlims = [lon[idx_gap_2], lon[idx_gap_1]]

    else:
        lonlims = [lon[lonidx[0]], lon[lonidx[-1]]]


    #setting up figures

    eofsolver, mask_ocean = EOF_ocean(APE, OB)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    eof = solver.eofsAsCovariance(neofs=neofs)
    pc = solver.pcs(npcs=neofs, pcscaling=1)
    
    df = pd.DataFrame()
    for e in range(neofs):
        df[e] = pc[:, e]
    rolling = df.rolling(12, center = True) #12 month rolling mean
    
    lonlims, latlims = find_lonlatlims(eof)
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
    fig = plt.figure(figsize = (8, 3), layout = 'tight')
    
    proj = find_proj(OB)
        
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        basin, lonb, latb = crop_oceanbasin(eof[n, :, :], lon, lat)
    else:
        basin, lonb, latb = eof[n, :, :], lon, lat

    ax_spatial = plt.subplot(121, projection = proj)
    ax_spatial.set_facecolor('darkgrey')
    ax_spatial.coastlines()
    basin_c, lon_c = add_cyclic_point(basin, coord=lonb, axis=-1)
    
    
    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    plot = ax_spatial.contourf(lon_c, latb, basin_c, 
                cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
    cbar = plt.colorbar(plot, ax = ax_spatial, location = cbarloc,
                 label = 'APE Variance, $Jm^{-2}$')
    ax_spatial.set_ylim(latlims[0], latlims[-1])

    
    ax_PC = plt.subplot(122)
    ax_PC.plot(time, pc[:, n], color = 'royalblue')
    ax_PC.plot(time, rolling.mean()[n], color = 'black')
    ax_PC.set_ylabel(f'PC{n+1}, arb')
    ax_PC.set_xlabel('Time')
    ax_PC.xaxis.set_minor_locator(mdates.YearLocator(10))
    ax_PC.xaxis.set_major_locator(mdates.YearLocator(20))
    
    ticks = cbar.get_ticks()
    if cbarloc in ['bottom', 'top']:
        if 0 in ticks[::2]:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=False)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=True)
        else:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=True)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=False)
    
    ax_spatial.set_extent(extent)
    ax_spatial.set_ylim(latlims[0], latlims[-1])
    gls = ax_spatial.gridlines(draw_labels = True, color = 'black', alpha = 0.2)
    gls.top_labels=False
    gls.right_labels=False 



np.save('SP_integrated_PC1', pc1[:, 0])
plt.savefig('SP_integrated_SP.pdf', bbox_inches= 'tight')