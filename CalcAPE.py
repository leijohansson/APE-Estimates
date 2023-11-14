# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:54:06 2023

@author: Linne
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
R = 6.371e6
#setting maximum depth to calculate to
max_depth = 700

#reading data to get shape
SP_eg = data.salinity.to_numpy().squeeze()

#converting lat, lon, z into same shape as data
lon1D = data.lon.to_numpy()
lon = np.zeros(SP_eg.shape)
for i in range(len(lon1D)):
    lon[:, :, i] = lon1D[i]

lat1D = data.lat.to_numpy()
lat = np.zeros(SP_eg.shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]

z1D = data.depth.to_numpy()
z = np.zeros(SP_eg.shape)
for i in range(len(z1D)):
    z[i, :, :] = z1D[i]

#calculating V_ijk, and formatting into the same size array as data
#same volume array used for potential temperature
depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
A_j = R**2 * np.pi/180*(np.sin((lat1D+0.5)*np.pi/180) - 
                         np.sin((lat1D-0.5)*np.pi/180))
A_ij = np.zeros((SP_eg.shape[1:]))
V_ijk = np.zeros((SP_eg.shape))
for i in range(len(lon1D)):
    A_ij[:, i] = A_j
for i in range(len(depths)):
    V_ijk[i, :, :] = (depth_bnds[i, 1] - depth_bnds[i, 0])*A_ij

#calculating p, function states that up is positive direction
p = gsw.conversions.p_from_z(-z, lat)

#restricting depths to max depth
ndepths = len(depths[depths<max_depth])
V_ijk = V_ijk[:ndepths, :, :]
lat, lon = lat[:ndepths, :, :], lon[:ndepths, :, :]
z, p = z[:ndepths, :, :], p[:ndepths, :, :]

#setting up arrays to record values
#setting length of arr to be number of files in data file
APE_arr = np.zeros((len(os.listdir('Data'))))
years = APE_arr.copy()
months = APE_arr.copy()
idx = -1	

for year in range(1950, 2024):
    print(year)
    start =time.time()
    for month in range(1, 13):
        idx+=1
        if idx == len(os.listdir('Data')):
            #stop loop if no more files to run
            break
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
        filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
        data = xr.open_dataset(f'{datadir}/{filename}')
        
        SP = data.salinity.to_numpy().squeeze()[:ndepths, :, :]
        #convert potential temperature fromo K to C
        PT = data.temperature.to_numpy().squeeze()[:ndepths, :, :] - 273.15
        
        #calculating reference salinity from practical salinity
        SR = gsw.conversions.SR_from_SP(SP)
        
        #calculating absolute salinity from practical salinity
        SA = gsw.conversions.SA_from_SP(SP, p, lon, lat)
        
        #calculating conservative temperature from abslute salinity and 
        #potential temperature
        CT = gsw.conversions.CT_from_pt(SA, PT)
        
        #calculating zref, pref
        gammat, zref, pref, sigref = gsw_gammat_analytic_CT_exact(SR, CT)
        #calculating enthalpy difference
        h_diff = gsw.energy.enthalpy_diff(SA, CT, pref, p)
        
        bigterm = h_diff + 9.81*(z-zref)
        
        rho = gsw.density.rho(SA, CT, p)
        
        #calculating APE for each grid point
        APE_dV = bigterm*rho*V_ijk
        #getting rid of nans
        APE_dV= np.nan_to_num(APE_dV)
        # validvol_bool = (APE_dV != 0).astype(int) #needed if dividing by volume
        #integrating over space
        APE = np.sum(APE_dV)
        APE_arr[idx] = APE
        years[idx] = year
        months[idx] = str(month)
    print('Time taken:',  time.time()-start, 's')
    
APEDf = pd.DataFrame()
APEDf['APE'] = APE_arr
APEDf['year']= years
APEDf['month'] = months
APEDf.to_csv('APE.csv')

import seaborn as sns
sns.set_context('talk')
plt.figure()
plt.title('APE (depth < 700m)')
plt.plot(APEDf['APE'])
plt.xticks(np.arange(0, len(APEdf), 120), labels = np.arange(1950, 2024, 10))
plt.xlabel('Time')
plt.ylabel(r'$APE$')
plt.savefig('APE_monthly_2.pdf', bbox_inches='tight')

