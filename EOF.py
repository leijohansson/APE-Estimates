# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:58:46 2024

@author: Linne

EOF
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

def make_EOFsolver(data, mask = None):
    mask3d = np.broadcast_to(mask, data.shape)    
    m_data= ma.masked_array(data, mask = mask3d)
    solver = Eof(m_data) #, weights=wgts)
    return solver

if __name__ == '__main__':
    with open('RegionFilters/ocean_filters-EN4.pkl', 'rb') as f:
        ocean_filters = pickle.load(f)
    
    startyear = 1960
    endyear = 2022
    nyears = endyear + 1 - startyear
    nmonths = nyears * 12
    x_time =  pd.date_range(f'{startyear}-01-01', periods=nmonths, freq='m')
    
    neofs = 1
    
    datadir = datapath + 'Data' 
    filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
    data = xr.open_dataset(f'{datadir}/{filename}')
    surface_s = data.salinity.values[0,0, :, :]
    surface_valid = surface_s/surface_s
    shape = data.salinity.squeeze().shape
    mask = np.nan_to_num(surface_valid-1, nan = 1)
    
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
            #calculating APE up to depth 700
            APE_700 = np.sum(APE_all[:24, :, :], axis = 0)
            mAPE = ma.masked_array(APE_700, mask = mask)
            timespace_arr[timeidx, :, :] = mAPE
    #%%
    fig, axs = plt.subplots(4, 2, figsize=(12, 15))
    # fig1, axs1 = plt.subplots(4, 2, figsize=(12, 15))
    
    f_i = 0
    lon, lat = data.lon.to_numpy(), data.lat.to_numpy()
    for OB in ocean_filters.keys():
        OF = ocean_filters[OB]
        masksum = np.isnan(OF) + mask
        mask_ocean = (masksum>0)
        solver = make_EOFsolver(timespace_arr, mask_ocean)
        
        eof1 = solver.eofsAsCovariance(neofs=neofs)
        pc1 = solver.pcs(npcs=1, pcscaling=1)
        fracs = solver.varianceFraction(neofs)
    
        basin, lonb, latb = crop_oceanbasin(eof1[0, :, :], lon, lat)
        plot = axs[f_i//2, f_i%2].imshow(basin, #levels=clevs,
                    cmap=plt.cm.RdBu_r)
        fig.colorbar(plot, ax = axs[f_i//2, f_i%2], shrink=0.6)
        axs[f_i//2, f_i%2].set_title(f'{OB} EOF1: {round(fracs[0], 4)*100}%')
    
        # axs[f_i//2, f_i%2]
        
        f_i += 1
        
    fig.tight_layout()
    
        #%%
    
    
    # plt.plot(pc1)
    
    # #%%
    fig, axs = plt.subplots(neofs, 1, figsize = (8, 10))
    for i in range(neofs):
        ax = axs[i]
        ax.set_facecolor('lightgrey')
        plot = ax.contourf(data.lon, data.lat, eof1[i, :, :], #levels=clevs,
                    cmap=plt.cm.RdBu_r)#, transform=ccrs.PlateCarree())
        ax.set_title(f'EOF{i}: ')
        plt.colorbar(plot)
    fig.suptitle('EOF expressed as covariance')
    plt.tight_layout()
    plt.show()
    plt.savefig('EOF_1960-2022.pdf', bbox_inches = 'tight')