# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:22:44 2024

@author: Linne
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import pickle
from FuncsAPE import crop_oceanbasin, datapath
from EN4_singledepth import EN4_singledepth_time

from EOF import make_EOFsolver
log = False
startyear = 1960
endyear = 2020
depth = 800
neofs = 1
cont = False

APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear)
true_depth = int(np.round(true_depth[0], -1))
extent = [lon[0], lon[-1], lat[0], lat[-1]]
log10 = np.log10(APE)

with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
    
datadir = datapath + 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s
shape = data.salinity.squeeze().shape
mask = np.nan_to_num(surface_valid-1, nan = 1)
    
    
fig, axs = plt.subplots(4, 2, figsize=(12, 15))
fig1, axs1 = plt.subplots(4, 2, figsize=(12, 15))

if log:
    fig.suptitle(f'EOF as Covariance: log10 APE density at {true_depth}m')
    fig1.suptitle(f'EOF PC1: log10 APE density at {true_depth}m')
else:
    fig.suptitle(f'EOF as Covariance: APE density at {true_depth}m')
    fig1.suptitle(f'EOF PC1: APE density at {true_depth}m')
f_i = 0
lon, lat = data.lon.to_numpy(), data.lat.to_numpy()
for OB in ocean_filters.keys():
    OF = ocean_filters[OB]
    masksum = np.isnan(OF) + mask
    mask_ocean = (masksum>0)
    try:
        if log:
            solver = make_EOFsolver(log10, mask_ocean)
        else:
            solver = make_EOFsolver(APE, mask_ocean)            
        eof1 = solver.eofsAsCovariance(neofs=neofs)
        pc1 = solver.pcs(npcs=1, pcscaling=1)
        fracs = solver.varianceFraction(neofs)
    
        basin, lonb, latb = crop_oceanbasin(eof1[0, :, :], lon, lat)
        plot = axs[f_i//2, f_i%2].imshow(basin, #levels=clevs,
                    cmap=plt.cm.RdBu_r)
        fig.colorbar(plot, ax = axs[f_i//2, f_i%2], shrink=0.6)
        axs[f_i//2, f_i%2].set_title(f'{OB} EOF1: {round(fracs[0], 4)*100}%')
        axs1[f_i//2, f_i%2].plot(time, pc1)

    except:
        axs[f_i//2, f_i%2].set_title(f'{OB}')


    axs1[f_i//2, f_i%2].set_title(f'{OB}')

    
    f_i += 1
    
fig.tight_layout()
fig1.tight_layout()


if log:
    fig.savefig(f'EN4 Plots/EOF_log_onedepth_{true_depth}_spatial.pdf', bbox_inches = 'tight')
    fig.savefig(f'EN4 Plots/EOF_log_onedepth_{true_depth}_PC1.pdf', bbox_inches = 'tight')
else: 
    fig.savefig(f'EN4 Plots/EOF_onedepth_{true_depth}_spatial.pdf', bbox_inches = 'tight')
    fig.savefig(f'EN4 Plots/EOF_onedepth_{true_depth}_PC1.pdf', bbox_inches = 'tight')