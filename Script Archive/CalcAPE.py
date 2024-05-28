# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:54:06 2023

@author: Linne

Old CalcAPE file
"""

import numpy as np
import gsw
from gsw_gammat_analytic_CT_exact import *
from gsw_gammat_analytic_CT import *
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

datadir = 'Data' 
#data file has all monthly files inside (no subfolders)
#nothing else in the data file

filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')

start_year = 2000
end_year = 2000
#reading data to get shape
V_ijk = np.load('dVolume.npy', allow_pickle=True)
#I have checked V_ijk for negative values

#converting lat, lon, z into same shape as data
lon1D = data.lon.to_numpy()
lon = np.zeros(V_ijk.shape)
for i in range(len(lon1D)):
    lon[:, :, i] = lon1D[i]

lat1D = data.lat.to_numpy()
lat = np.zeros(V_ijk.shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]

z1D = data.depth.to_numpy()
z = np.zeros(V_ijk.shape)
for i in range(len(z1D)):
    z[i, :, :] = z1D[i]

#calculating p, function states that up is positive direction
p = gsw.conversions.p_from_z(-z, lat)

#restricting depths to max depth
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()

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
        #convert potential temperature from K to C
        PT = data.temperature.to_numpy().squeeze() - 273.15
        
        #calculating reference salinity from practical salinity
        SR = gsw.conversions.SR_from_SP(SP)
        #calculating conservative temperature from SR and 
        #potential temperature
        CT = gsw.conversions.CT_from_pt(SR, PT)

        #calculating zref, pref
        gammat, zref, pref, sigref = gsw_gammat_analytic_CT_exact(SR, CT)
        #calculating enthalpies
        href = gsw.energy.enthalpy(SR, CT, pref)
        # href = np.where(href<0, np.nan, href) #getting rid of all negative values
        h = gsw.energy.enthalpy(SR, CT, p)
        # h= np.where(h<0, np.nan, h) #getting rid of all negative values 
        
        rho = gsw.density.rho(SR, CT, p)
        
        #background energy
        BGE = (href-9.81*zref)*rho*V_ijk
        BGE= np.nan_to_num(BGE)
        
        #total energy
        TE = (h-9.81*z)*rho*V_ijk
        TE = np.nan_to_num(TE)
        
        #APE
        APE_dV = TE-BGE
        APE_dV= np.nan_to_num(APE_dV)

        
        #Alternative calculation of APE
        # h_diff = h-href
        # bigterm = h_diff - 9.81*(z-zref)
        # APE_dV = bigterm*rho*V_ijk
        # APE_dV= np.nan_to_num(APE_dV)
        
        # np.save(f'BGEarrays/BGE_{year}-{month}.npy', BGE)
        # np.save(f'APEarrays/APE_{year}-{month}.npy', APE_dV)
        print('APE all depths:', np.sum(APE_dV))
        print('APE <700m:', np.sum(APE_dV[:24, :, :]))

    print('Time taken:',  time.time()-start, 's')
    
