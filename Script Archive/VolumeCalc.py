# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:42:32 2023

@author: Linne

Old File for calculating V_ijk array for EN4
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
start_year = 2010
end_year = 2023
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

dgrid = 1
#calculating V_ijk, and formatting into the same size array as data
#same volume array used for potential temperature
depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
A_j = R**2 * np.pi/180*(np.sin((lat1D+dgrid/2)*np.pi/180) - 
                         np.sin((lat1D-dgrid/2)*np.pi/180))
A_ij = np.zeros((SP_eg.shape[1:]))
V_ijk = np.zeros((SP_eg.shape))
for i in range(len(lon1D)):
    A_ij[:, i] = A_j
for i in range(len(depths)):
    V_ijk[i, :, :] = (depth_bnds[i, 1] - depth_bnds[i, 0])*A_ij
np.save('dVolume', V_ijk)
#%%
def calc_Aij(data):
    R = 6.371e6
    shape = data.salinity.to_numpy().squeeze().shape

    lat1D = data.lat.to_numpy()

    dgrid = lat1D[1] - lat1D[0]
    print(dgrid)
    A_j = dgrid * R**2 * np.pi/180*(np.sin((lat1D+dgrid/2)*np.pi/180) - 
                            np.sin((lat1D-dgrid/2)*np.pi/180))
    A_ij = np.zeros(shape[1:])
    for i in range(len(data.lon)):
        A_ij[:, i] = A_j

    return A_ij