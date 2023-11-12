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
import xarray as xr
import pandas as pd

datadir = 'Data'
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
R = 6.371e6
max_depth = 700

SP_eg = data.salinity.to_numpy().squeeze()

lon1D = data.lon.to_numpy()
lat1D = data.lat.to_numpy()
z1D = data.depth.to_numpy()
lat = np.zeros(SP_eg.shape)
lon = np.zeros(SP_eg.shape)
z = np.zeros(SP_eg.shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]
for i in range(len(lon1D)):
    lon[:, :, i] = lon1D[i]
for i in range(len(z1D)):
    z[i, :, :] = z1D[i]
    
depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
T = data.temperature.to_numpy().squeeze()
A_j = R**2 * np.pi/180*(np.sin((lat1D+0.5)*np.pi/180) - 
                         np.sin((lat1D-0.5)*np.pi/180))
A_ij = np.zeros((T.shape[1:]))
V_ijk = np.zeros((T.shape))
for i in range(len(lon1D)):
    A_ij[:, i] = A_j
for i in range(len(depths)):
    V_ijk[i, :, :] = (depth_bnds[i, 1] - depth_bnds[i, 0])*A_ij

p = gsw.conversions.p_from_z(-z, lat)


ndepths = len(depths[depths<max_depth])
V_ijk = V_ijk[:ndepths, :, :]
lat, lon, z, p = lat[:ndepths, :, :], lon[:ndepths, :, :], z[:ndepths, :, :], p[:ndepths, :, :]
APE_arr = np.zeros((len(os.listdir('Data'))))
years = APE_arr.copy()
months = APE_arr.copy()
idx = -1

#looping through years and files
for year in range(1950, 1951):
    print(year)
    for month in range(1, 13):
        idx+=1
        if len(str(month)) == 1:
            month = '0'+str(month)
        filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
        data = xr.open_dataset(f'{datadir}/{filename}')
        
        #reading dataa
        SP = data.salinity.to_numpy().squeeze()[:ndepths, :, :]
        PT = data.temperature.to_numpy().squeeze()[:ndepths, :, :] - 273.15
        
        #gsw conversions to get SR and CT
        SR = gsw.conversions.SR_from_SP(SP)
        SA = gsw.conversions.SA_from_SP(SP, p, lon, lat)
        CT = gsw.conversions.CT_from_pt(SA, PT)
        
        #finding zref
        gammat, zref, pref, sigref = gsw_gammat_analytic_CT_exact(SR, CT)
        
        #calculating APE
        p_zref = gsw.conversions.p_from_z(-zref, lat)
        # h_z = gsw.energy.enthalpy(SA, CT, p)
        # h_zref = gsw.energy.enthalpy(SA, CT, p_zref)
        h_diff = gsw.energy.enthalpy_diff(SA, CT, p, p_zref)
        bigterm = h_diff + 9.81*(zref-z)

        rho = gsw.density.rho(SA, CT, p)
        APE_dV = bigterm*rho*V_ijk
        APE_dV= np.nan_to_num(APE_dV)
        validvol_bool = (APE_dV != 0).astype(int)
        APE = np.sum(APE_dV)/np.sum(V_ijk*validvol_bool)
        APE_arr[idx] = APE
        years[idx] = year
        months[idx] = str(month)
        
APEDf = pd.DataFrame()
APEf['T'] = APE_arr
APEDf['year']= years
APEDf['month'] = months
APEDf.to_csv('APE.csv')


