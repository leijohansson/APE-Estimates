# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:38:32 2024

@author: Linne
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from FuncsAPE import *
import gsw



datadir = datapath + 'Data' 
#data file has all monthly files inside (no subfolders)
#nothing else in the data file

filename = 'EN.4.2.2.f.analysis.g10.196001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
start_year = 1960
end_year = 2023

shape = data.salinity.squeeze().shape
#converting lat, z into same shape as data
lat1D = data.lat.to_numpy()
lat = np.zeros(shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]

lon1D = data.lon.to_numpy()
lon = np.zeros(shape)
for i in range(len(lon1D)):
    lat[:, :, i] = lon1D[i]


z1D = data.depth.to_numpy()
z = np.zeros(shape)
for i in range(len(z1D)):
    z[i, :, :] = z1D[i]

depth_bnds = data.depth_bnds.to_numpy()
A_ij = calc_Aij(data)
V_ijk = np.zeros((shape))
for i in range(len(depth_bnds)):
    V_ijk[i, :, :] = (depth_bnds[i, 1] - depth_bnds[i, 0])*A_ij

#calculating p, function states that up is positive direction
# p = gsw.conversions.p_from_z(-z, lat)
# RGJT: Pressure needs to be defined in terms of Lorenz reference pressure pr(z) for local APE density
# to be positive definite 
p = pr(z) 
#restricting depths to max depth

timeseries = np.zeros((end_year+1-start_year)*12)
#%%
i = 0
for year in range(start_year, end_year+1):
    print(year)
    start =time.time()
    for month in range(1, 13):
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
        filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
        data = xr.open_dataset(f'{datadir}/{filename}')
        
        SP = data.salinity.to_numpy().squeeze()
        # convert potential temperature from K to C
        PT = data.temperature.to_numpy().squeeze() - 273.15

        SA = gsw.SA_from_SP(SP, p, lon, lat)
        SR = gsw.conversions.SR_from_SP(SP)

        CT = gsw.conversions.CT_from_pt(SR, PT)

        
        pot_h = gsw.energy.enthalpy(SR, PT, 0)
        rho = gsw.density.rho(SR, CT, p)
        
        
        rho_int = pot_h * rho * V_ijk
        timeseries[i] = np.nansum(rho_int)
        i += 1
plt.plot(timeseries)
#%%
np.save(f'potential_enthalpy_{start_year}-{end_year}', timeseries)


        