# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:07:55 2023

@author: Linne
Checking area of globe
"""

import gsw
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
datadir = 'Data'
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
max_depth = 700

R = 6371e3

depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
lats = np.arange(0, 180)
lons = np.arange(0, 360)
T = data.temperature.to_numpy().squeeze()
A_j = R**2 * np.pi/180*(np.sin((lats+0.5)*np.pi/180) - 
                        np.sin((lats-0.5)*np.pi/180))
A_ij = np.zeros((180, 360))
for i in range(len(lons)):
    A_ij[:, i] = A_j
print(np.sum(A_ij)/1e6)