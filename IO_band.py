# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 18:13:45 2024

@author: Linne
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from FuncsAPE import datapath, calc_Aij
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

def lin_regress_subsetyears(startyear = 0, endyear = np.inf, ax = None, alternative = 'two-sided'):
    bool1 = (x_time.year<=endyear).astype(int)
    bool2 = (x_time.year>=startyear).astype(int)
    time_sel = np.where((bool1+bool2) == 2)
    
    slope, intercept, r, p, std_err = stats.linregress((np.arange(len(x_time))/12)[time_sel], TS_APE[time_sel], alternative = alternative
                                                       )
    print(f'{startyear}-{endyear}, Trend : {slope/1e17} +/- {std_err/1e17} x 10^17 J/year,')
    print(f'R2 value: {r}, p_value: {p}')
    if ax: 
        ax.plot(x_time[time_sel], slope*(np.arange(len(x_time))/12)[time_sel] + intercept, lw = '2', color = 'red', label = 'Trend Line')


asfrac = False
startyear = 1960
endyear = 2023


with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
    
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    IO30 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
    
IO_band = IO45-IO30

nmonths = (endyear + 1 - startyear)*12
x_time =  pd.date_range(f'{startyear}-01-01', periods=nmonths, freq='m')

def plot_TS():
    TS_BGE = np.zeros(nmonths)
    TS_APE = TS_BGE.copy()
    time_id = -1
    for year in range(startyear, endyear+1):
        for month in range(1, 13):
            time_id += 1
            if len(str(month)) == 1:
                #eg '1' becomes '01' (as in the filenames)
                month = '0'+str(month)
            
            file = f'\APEarrays\APE_{year}-{month}.npy'
            APE_all = np.load(datapath + file)
            #calculating APE up to depth 700
            APE = np.sum(APE_all, axis = 0)
            
            APE_ocean = np.nansum(APE * IO_band)
            TS_APE[time_id] = APE_ocean
            
            file = f'\BGEarrays\BGE_{year}-{month}.npy'
            BGE_all = np.load(datapath + file)
            #calculating APE up to depth 700
            BGE = np.sum(BGE_all, axis = 0)
        

            BGE_ocean = np.nansum(BGE * IO_band)
            TS_BGE[time_id] = BGE_ocean

    rolling_n = 12
    #creating dataframe
    df_APE, df_BGE = pd.DataFrame(TS_APE, columns=['APE']), pd.DataFrame(TS_BGE, columns = ['BGE'])
    rolling_APE = df_APE.rolling(rolling_n, center = True)
    rolling_BGE= df_BGE.rolling(rolling_n, center = True)
    
    fig, axs = plt.subplots(2,  1, sharex = True)
    fig.suptitle('Indian Ocean, 30S-45S')
    
    # axs[0].set_title('Available Potential Energy')
    # axs[0].set_ylabel('Volume Integrated APE, $J$')
    # axs[0].plot(x_time, TS_APE, alpha = 0.6, label = 'Data')
    # axs[0].plot(x_time, rolling_APE.mean()['APE'], color = 'black', label = f'{rolling_n} month rolling mean')
    
    # lin_regress_subsetyears(2003, np.inf, axs[0])
    # axs[0].legend()
    # lin_regress_subsetyears(1960, 2000, axs[0])
    
    
    # axs[1].set_title('Background Potential Energy')
    # axs[1].set_ylabel('Volume Integrated BGE, $J$')
    # axs[1].plot(x_time, TS_BGE, alpha = 0.6, label = 'Data')
    # axs[1].plot(x_time, rolling_BGE.mean()['BGE'], color = 'black', label = f'{rolling_n} month rolling mean')
    # axs[1].set_xlabel('Time')
    
    
    # plt.savefig(f'EN4 Plots/TS Plots/BGE_APE_TS_IOBand.png', bbox_inches = 'tight')
    
    
    OB = 'Indian Ocean'
    bool1 = (x_time.year<=2000).astype(int)
    bool2 = (x_time.year>=1960).astype(int)
    time_sel = np.where((bool1+bool2) == 2)
    before = np.mean(TS_APE[time_sel])

    bool1 = (x_time.year<=np.inf).astype(int)
    bool2 = (x_time.year>=2003).astype(int)
    time_sel = np.where((bool1+bool2) == 2)
    after = np.mean(TS_APE[time_sel])

    print(OB)
    print('Change:', after-before, 'J')
    print('% Change:', (after-before)/np.mean(TS_APE)*100, '%')
    
plot_TS()





