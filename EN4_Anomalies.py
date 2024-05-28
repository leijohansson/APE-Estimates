# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:07:47 2024

@author: Linne

Plot vertically integrated EN4 anomalies
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from FuncsAPE import find_depthfracs

max_depth= 700
startyear = 2010
endyear = 2019


datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
surface_s = data.salinity.values[0,0, :, :]
surface_valid = surface_s/surface_s

#making array to account for maximum depth
depth_bnds = data.depth_bnds.to_numpy()
dz = depth_bnds[:, 1] - depth_bnds[:, 0]
depth_fracs = find_depthfracs(dz, data.salinity.squeeze().shape, max_depth)

#getting shape of vertically integrated APE array
year = 2010
month = '05'
APE = np.load(f'APEarrays/APE_{year}-{month}.npy')
APE_int_700 = np.sum(APE*depth_fracs, axis = 0)

#calculating mean APE, and monthly mean APE
APE_months_list = []
for month in range(1, 13):
    month_sum = np.zeros(APE_int_700.shape)
    if len(str(month)) == 1:
        #eg '1' becomes '01' (as in the filenames)
        month = '0'+str(month)
    for year in range(startyear, endyear + 1):
        APE = np.load(f'APEarrays/APE_{year}-{month}.npy')
        month_sum += np.sum(APE*depth_fracs, axis = 0)
    APE_months_list.append(month_sum/(endyear+1-startyear))
    
#find average APE over whole period
APE_sum = np.zeros(APE_int_700.shape)
for i in APE_months_list:
    APE_sum += i
APE_ave = APE_sum/12

#plotting each mmonth
for i in range(12):
    month = i+1
    if len(str(month)) == 1:
        month = '0'+str(month)
    fig, ax = plt.subplots()
    ax.set_facecolor('lightgrey')
    plot = ax.imshow(np.flip((APE_months_list[i] - APE_ave)*
                             surface_valid, axis = 0), vmax = 4e16, 
                     vmin = -4e16, cmap = 'seismic')
    ax.set_title(r'$APE - \bar{APE}$'+ f' Month: {month}, {startyear}-{endyear} mean')
    ax.set_ylabel('Latitude, $^\circ$')
    ax.set_xlabel('Longitude, $^\circ$')
    fig.colorbar(plot, ax= ax,label = 'Vertically Integrated APE anomaly, $Jm^{-2}$', 
                 location = 'bottom')
    
    fig.savefig('EN4 Plots/VI_anomaly_{month}.png')
    


