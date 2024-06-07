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
#whether to use log10 of density at depth
log = False
#if density == True, take APE in J/kg, else take in J/m3
density = False
#setting time boundaries
startyear = 1960
endyear = 2020

#setting depth to choose
depth = 800

#setting number of eofs to plot
neofs = 4

#reading in and taking data only at a single depth
APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density)
true_depth = int(np.round(true_depth[0], -1))
extent = [lon[0], lon[-1], lat[0], lat[-1]]

if log:
    log10 = np.log10(APE)

#reading in filters for different ocean basins
with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
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

#looping through ocean basins
for OB in ocean_filters.keys():
    
    #combining filters with land masks to make combined mask
    OF = ocean_filters[OB]
    masksum = np.isnan(OF) + mask
    mask_ocean = (masksum>0)
    
    #setting up figures
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        fig, axs = plt.subplots(neofs//2, 2, figsize=(18, 4))
    else:
        fig, axs = plt.subplots(neofs//2, 2, figsize=(12, 12))

    fig1, axs1 = plt.subplots(neofs//2, 2, figsize=(12, 15))

    
    if log:
        fig.suptitle(f'{OB}, EOF as Covariance: log10 APE density at {true_depth}m')
        fig1.suptitle(f'{OB}, EOF PC1: log10 APE density at {true_depth}m')
    else:
        fig.suptitle(f'{OB}, EOF as Covariance: APE density at {true_depth}m')
        fig1.suptitle(f'{OB}, EOF PC1: APE density at {true_depth}m')
        
    # try:
    if log:
        solver = make_EOFsolver(log10, mask_ocean)
    else:
        solver = make_EOFsolver(APE, mask_ocean)            
    eof1 = solver.eofsAsCovariance(neofs=neofs)
    pc1 = solver.pcs(npcs=neofs, pcscaling=1)
    fracs = solver.varianceFraction(neofs)
    for e in range(neofs):
        basin, lonb, latb = crop_oceanbasin(eof1[e, :, :], lon, lat)
        
        plot = axs[e//2, e%2].imshow(basin, #levels=clevs,
                    cmap=plt.cm.RdBu_r)
        fig.colorbar(plot, ax = axs[e//2, e%2], shrink=0.8, location = 'bottom')        
        axs1[e//2, e%2].plot(time, pc1[:, e])
        
        axs[e//2, e%2].set_title(f'EOF {int(e+1)}: {round(fracs[e], 4)*100}%')
        axs1[e//2, e%2].set_title(f'PC {int(e+1)}: {round(fracs[e], 4)*100}%')

        
    fig.tight_layout()
    fig1.tight_layout()
    OB = OB.replace(" ", "") #removing space for filename purposes
    
    if log:
        fig.savefig(f'EN4 Plots/EOF_log_onedepth_{OB}_{true_depth}_spatial.pdf', bbox_inches = 'tight')
        fig.savefig(f'EN4 Plots/EOF_log_onedepth_{OB}_{true_depth}_PC1.pdf', bbox_inches = 'tight')
    else: 
        fig.savefig(f'EN4 Plots/EOF_onedepth_{true_depth}_spatial.pdf', bbox_inches = 'tight')
        fig.savefig(f'EN4 Plots/EOF_onedepth_{true_depth}_PC1.pdf', bbox_inches = 'tight')
        


    # except:
        # print(f'{OB} didnt work. investigate me pls')



    
    
