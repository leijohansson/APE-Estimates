# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:00:51 2024

@author: Linne

'Time series' (Jan-Dec) of WOCE APE climatologies
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from FuncsAPE import *
#grid spacing 0.25 deg
import scipy.io
import numpy.ma as ma
import pickle 
import seaborn as sns

SO_cutoff_lat = -45
palette = sns.crayon_palette(['Brick Red', 'Denim'])
sns.set_theme(context='notebook', style='white', palette = palette, 
              rc={'xtick.bottom': True,'ytick.left': True,})


datadir = datapath + 'WOCE_Data/Data/'
method = 'BAR'
month = '01'
filename = f'WAGHC_{method}_{month}_UHAM-ICDC_v1_0_1.nc'
data = xr.open_dataset(datadir+filename)
shape = data.temperature.squeeze().shape

depths = data.depth.to_numpy()
dz = np.zeros(len(depths))
dz[0] = depths[1]/2
depth_sum = dz[0]
for i in range(1, len(depths)):
   dz_i = (depths[i]- depth_sum)*2
   dz[i] = dz_i
   depth_sum += dz_i
   
# depth_fracs = find_depthfracs(dz, shape, max_depth)

plt.figure(figsize = (8, 3))
# plt.title('APE Climatology')

months = np.zeros(12)
for i in range(12):
    month = i+1
    if len(str(month)) == 1:
        #eg '1' becomes '01' (as in the filenames)
        month = '0'+str(month)
    months[i] = month

i=0
for method in ['BAR', 'PYC']:
    APE_700= np.zeros(12)
    timeidx = -1
    for month in range(1, 13):
        timeidx += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)

        file = datapath + f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        APE_700[timeidx]= np.sum(APE_all)
    plt.plot(months, APE_700, label = method, color = palette[i])
    i+= 1

plt.xlabel('Month')
plt.ylabel('Volume Integrated APE, J')
plt.legend()
plt.savefig('WOCE Plots/APE_Climatology_lineplot.pdf', bbox_inches = 'tight')

#%%

#calculating and plotting volume integrated APE for different ocean basins
with open(f'RegionFilters/ocean_filters-WOCE_{SO_cutoff_lat}.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)

dictkey = list(ocean_filters.keys())[0]
ocean_filters['World'] = np.ones(ocean_filters[dictkey].shape)

n_months = 12

oceans = ['North Atlantic Ocean', 'South Atlantic Ocean',
          'North Pacific Ocean', 'South Pacific Ocean',
          'Indian Ocean', 'Southern Ocean', 'Arctic Ocean',
          'Mediterranean Region']#, 'Baltic Sea',
           # 'South China and Easter Archipelagic Seas']

methods_dict = {}

fig, axs = plt.subplots(4, 2, figsize=(10, 7), sharex = True)

i = 0
for method in ['BAR', 'PYC']:
    time_id = -1
    TS_oceans = {}
    for OB in ocean_filters.keys():
        TS_oceans[OB] = np.zeros((n_months))
        
    for month in range(1, 13):
        time_id += 1
        if len(str(month)) == 1:
            #eg '1' becomes '01' (as in the filenames)
            month = '0'+str(month)
            
        file = datapath+f'WOCE_Data/APEarrays/WAGHC_APE_{method}-{month}.npy'
        APE_all = np.load(file)
        #calculating APE up to depth 700
        APE_700 = np.sum(APE_all, axis = 0)
        
        for OB in ocean_filters:
            APE_ocean = np.nansum(APE_700 * ocean_filters[OB])
            TS_oceans[OB][time_id] = APE_ocean


    f_i = 0
    for OB in oceans:
        axs[f_i//2, f_i%2].plot(months,TS_oceans[OB], label = method, color = palette[i])
        axs[f_i//2, f_i%2].set_title(OB)
    
        f_i += 1
    i += 1
    methods_dict[method] = TS_oceans

fig.text(-0.025, 0.5, 'Volume Integrated APE, $J$', ha='center', va='center', rotation='vertical')
axs[3, 0].set_xlabel('Month')
axs[3, 1].set_xlabel('Month')
# axs[2, 1].set_xticks(np.arange(2, 13, 2), labels = np.arange(2, 13, 2).astype(str))
axs[0, 1].legend(loc = 'upper left')
# axs[3, 1].set_visible(False)
plt.tight_layout()
fig.savefig('WOCE Plots/Ocean_Climatologies.pdf', bbox_inches = 'tight')
        
#%%
if 'World' not in oceans:
    oceans.append('World')
for OB in oceans:
    BAR = methods_dict['BAR'][OB]
    PYC = methods_dict['PYC'][OB]
    
    BARmean = np.mean(BAR)
    
    BARorder = np.log10(BARmean)
    BARorder = int(np.round(BARorder, 1))
    BARval = np.round(BARmean/10**BARorder, 3)
    
    PYCmean = np.mean(PYC)
    
    PYCorder = np.log10(PYCmean)
    PYCorder = int(np.round(PYCorder, 1))
    PYCval = np.round(PYCmean/10**PYCorder, 3)

    
    
    meandiff = np.mean(PYC-BAR)
    difforder = np.log10(np.abs(meandiff))
    difforder = int(np.round(difforder, 1))
    diffval = np.round(meandiff/10**difforder, 3)
    
    perc = np.round(meandiff/PYCmean*100, 3)
    
    
    
    # print(f'{OB} & {PYCval}'+ r' $\times 10^' + f'{PYCorder}$ & ${BARval}' + r' \times 10^'+ f'{BARorder}$ & ${diffval}'+r' \times 10^ '+f'{difforder}$& {perc}'+r' \\ ')
    print(OB, PYC.max() - PYC.min())
    
    
    
    
    
    
    
    
    