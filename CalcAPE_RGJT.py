# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:54:06 2023

@author: Linne

Calculate and save BGE and APE arrays from EN4 data
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from FuncsAPE import *


datadir = 'Data' 
#data file has all monthly files inside (no subfolders)
#nothing else in the data file

filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
start_year = 1960
end_year = 2022

shape = data.salinity.squeeze().shape
#converting lat, z into same shape as data
lat1D = data.lat.to_numpy()
lat = np.zeros(shape)
for i in range(len(lat1D)):
    lat[:, i, :] = lat1D[i]

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
dz = depth_bnds[:, 1] - depth_bnds[:, 0]

for year in range(start_year, end_year+1):
    print(year)
    start =time.time()
    for month in range(1, 13):
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
        filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
        data = xr.open_dataset(f'{datadir}/{filename}')
        
        BGE, APE_dV = calc_APE(datadir, filename, V_ijk, p, z)
        
        
        #calculation without function: remove at some point but here just in case
        # SP = data.salinity.to_numpy().squeeze()
        # #convert potential temperature from K to C
        # PT = data.temperature.to_numpy().squeeze() - 273.15
        
        # #calculating reference salinity from practical salinity
        # SR = gsw.conversions.SR_from_SP(SP)
        # #calculating conservative temperature from SR and 
        # #potential temperature
        # CT = gsw.conversions.CT_from_pt(SR, PT)

        # #calculating zref, pref
        # gammat, zref, pref, sigref = gsw_gammat_analytic_CT_exact(SR, CT)
        # #calculating enthalpies
        # href = gsw.energy.enthalpy(SR, CT, pref)
        # # href = np.where(href<0, np.nan, href) #getting rid of all negative values
        # h = gsw.energy.enthalpy(SR, CT, p)
        # # h= np.where(h<0, np.nan, h) #getting rid of all negative values 
        
        # rho = gsw.density.rho(SR, CT, p)
        
        # #background energy
        # BGE = (href-grav*zref)*rho*V_ijk
        # BGE= np.nan_to_num(BGE)
        
        # #total energy
        # TE = (h-grav*z)*rho*V_ijk
        # TE = np.nan_to_num(TE)
        
        # # APE density (from Tailleux 2018) = APE_dV/(rho*V_ijk) 
        # Pi2 = h - href - grav*(z-zref) 
        
        # #APE
        # APE_dV = TE-BGE
        # APE_dV= np.nan_to_num(APE_dV)

        
        #Alternative calculation of APE
        # h_diff = h-href
        # bigterm = h_diff - 9.81*(z-zref)
        # APE_dV = bigterm*rho*V_ijk
        # APE_dV= np.nan_to_num(APE_dV)
        

        np.save(f'BGEarrays/BGE_{year}-{month}.npy', BGE)
        np.save(f'APEarrays/APE_{year}-{month}.npy', APE_dV)
        
        print('APE all depths:', np.sum(APE_dV))

    print('Time taken:',  time.time()-start, 's')
    