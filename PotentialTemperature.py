# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:50:04 2023

@author: Linne
"""

import gsw
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import pandas as pd
datadir = 'Data'
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
max_depth = 700

R = 6.371e6

depth_bnds = data.depth_bnds.to_numpy()
depths = data.depth.to_numpy()
lats = data.lat.to_numpy()
lons = data.lon.to_numpy()
T = data.temperature.to_numpy().squeeze()
A_j = R**2 * np.pi/180*(np.sin((lats+0.5)*np.pi/180) - 
                         np.sin((lats-0.5)*np.pi/180))
A_ij = np.zeros((T.shape[1:]))
V_ijk = np.zeros((T.shape))
for i in range(len(lons)):
    A_ij[:, i] = A_j
    
for i in range(len(depths)):
    V_ijk[i, :, :] = (depth_bnds[i, 1] - depth_bnds[i, 0])*A_ij

idx = -1
pot_temps = np.zeros((len(os.listdir('Data'))))
years = pot_temps.copy()
months = pot_temps.copy()

ndepths = len(depths[depths<max_depth])
V_ijk = V_ijk[:ndepths, :, :]
for year in range(1950, 2024):
    print(year)
    for month in range(1, 13):
        idx += 1
        if idx == len(pot_temps):
            break
        if len(str(month)) == 1:
            month = '0'+str(month)
        filename = f'EN.4.2.2.f.analysis.g10.{year}{month}.nc'
        data = xr.open_dataset(f'{datadir}/{filename}')
        T = data.temperature.to_numpy().squeeze()[:ndepths, :, :] -  273.15
        T_V = T*V_ijk
        T_V= np.nan_to_num(T_V)
        validvol_bool = (T_V != 0).astype(int)
        pot_temp = np.sum(T_V)/np.sum(V_ijk*validvol_bool)
        pot_temps[idx] = pot_temp
        years[idx] = year
        months[idx] = str(month)
PotTempDf = pd.DataFrame()
PotTempDf['T'] = pot_temps
PotTempDf['year']= years
PotTempDf['month'] = months
PotTempDf.to_csv('PotentialTemperature.csv')
#%%
PotTempDf = pd.read_csv('PotentialTemperature.csv')
import seaborn as sns
sns.set_context('talk')
plt.figure()
plt.title('Potential Temperature (depth < 700m)')
plt.plot(PotTempDf['T'])
plt.xticks(np.arange(0, len(PotTempDf), 120), labels = np.arange(1950, 2024, 10))
plt.xlabel('Time')
plt.ylabel(r'$\Theta, ^\circ C$')
plt.savefig('PotTemp_monthly.pdf', bbox_inches = 'tight')

ave_PT = np.zeros(len(PotTempDf['year'].unique())-1)

for year in PotTempDf['year'].unique():
    if year != 2023:
        ave_PT[int(year) - 1950] = PotTempDf[PotTempDf['year'] == year]['T'].mean()
#%%
plt.figure()
plt.title('Potential Temperature (depth < 700m)')
plt.plot(ave_PT)
plt.xticks(np.arange(0, len(ave_PT), 10), labels = np.arange(1950, 2024, 10))
plt.xlabel('Time')
plt.ylabel(r'$\Theta, ^\circ C$')
plt.savefig('PotTemp_yearlyave.pdf', bbox_inches = 'tight')


