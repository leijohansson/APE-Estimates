# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:22:44 2024

@author: Linne
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import xarray as xr
import pickle
from FuncsAPE import crop_oceanbasin, datapath
from EN4_singledepth import EN4_singledepth_time
import os
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import geopandas as gpd
from EOF import make_EOFsolver
from cartopy.util import add_cyclic_point
from scipy.signal import welch

import matplotlib.path as mpath
countries = gpd.read_file(datapath+'ne_110m_land.zip')

#from https://nordicesmhub.github.io/NEGI-Abisko-2019/training/example_NorthPolarStereo_projection.html
def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

if __name__ == '__main__':
    #whether to use log10 of density at depth
    log = False
    #if density == True, take APE in J/kg, else take in J/m3
    density = True
    #setting time boundaries
    startyear = 1960
    endyear = 2023
    spectral = True
    
    
    #setting depth to choose
    # depth = 400
    #setting maximum (northmost) latitude for Southern Ocean. 
    #Also the southern most latitude for SPO, SAO and IO
    SO_cutoff_lat = -45
    #setting number of eofs to plot
    neofs = 4
    
    with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'rb') as f:
        ocean_filters = pickle.load(f)
    with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
        IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
    with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
        IO30 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
    IO_band = IO45-IO30
    IO_band[np.where(IO_band == 0)] = np.nan
    ocean_filters['IO Band'] = IO_band
    
    
        
    #reading in example data to make masks
    datadir = datapath + 'Data' 
    filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
    data = xr.open_dataset(f'{datadir}/{filename}')
    surface_s = data.salinity.values[0,0, :, :]
    surface_valid = surface_s/surface_s
    shape = data.salinity.squeeze().shape
    mask = np.nan_to_num(surface_valid-1, nan = 1)
    
    #getting lon and lat values for plotting
    lon, lat = data.lon.to_numpy(), data.lat.to_numpy()
    # lon[-1] = 0
    LON, LAT = np.meshgrid(lon, lat)
        
    for depth in [10]:
        plotdir = f'EN4 Plots/EOF_{depth}m'
        if f'EOF_{depth}m' not in os.listdir(path = 'EN4 Plots/'):
            os.mkdir(f'EN4 Plots/EOF_{depth}m')
        
        #reading in and taking data only at a single depth
        APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
        true_depth = int(np.round(true_depth[0], -1))
        extent = [lon[0], lon[-1], lat[0], lat[-1]]
        
        if log:
            log10 = np.log10(APE)
        
        #reading in filters for different ocean basins
    
        #%%
        #looping through ocean basins
        # for OB in ocean_filters.keys():
        # for OB in ['North Atlantic Ocean', 'South Atlantic Ocean']:
        # for OB in ['Arctic Ocean', 'Southern Ocean']:
        for OB in ['Arctic Ocean']:
            #%%
            OF = ocean_filters[OB]
            masksum = np.isnan(OF) + mask
            mask_ocean = (masksum>0)
            
            lon_m = ma.masked_array(LON, mask = mask_ocean)
            lat_m = ma.masked_array(LAT, mask = mask_ocean)
        
            
            
            if log:
                try:
                    solver = make_EOFsolver(log10, mask_ocean)
                except:
                    solver = 0
            else:
                try:
                    solver = make_EOFsolver(APE, mask_ocean)       
                except:
                    solver = 0
            #combining filters with land masks to make combined mask
            
                
        
            if type(solver) != int:
                eof1 = solver.eofsAsCovariance(neofs=neofs)
                pc1 = solver.pcs(npcs=neofs, pcscaling=1)
                fracs = solver.varianceFraction(neofs)
                # basin, lonb, latb = crop_oceanbasin(eof1[0, :, :], lon, lat)
                
                
                lon_where_1d = np.sum(eof1[0, :, :]<np.inf, axis = 0)
                lonidx = np.where(lon_where_1d >0)[0]
                lat_where_1d = np.sum(eof1[0, :, :]<np.inf, axis = 1)
                latidx = np.where(lat_where_1d >0)[0]
                latlims = [lat[latidx[0]], lat[latidx[-1]]]
        
                if True in ((lonidx[1:] - lonidx[0:-1]) > 100):
                    idx_gap_1 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]]
                    idx_gap_2 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]+1]
        
                    lonlims = [lon[idx_gap_2], lon[idx_gap_1]]
        
                else:
                    lonlims = [lon[lonidx[0]], lon[lonidx[-1]]]
            
        
                #setting up figures
            
                if OB == 'Arctic Ocean':
                    fig, axs = plt.subplots(neofs//2, 2, figsize=(10, 10),
                                            subplot_kw={'projection': ccrs.NorthPolarStereo()})
                elif OB == 'Southern Ocean':
                    fig, axs = plt.subplots(neofs//2, 2, figsize=(10, 10),
                                            subplot_kw={'projection': ccrs.SouthPolarStereo()})
                elif 'Pacific' in OB:
                    fig, axs = plt.subplots(neofs//2, 2, figsize=(12, 12),
                                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude = 180)})
        
                else:
                    fig, axs = plt.subplots(neofs//2, 2, figsize=(12, 12),
                                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
                    
            
                fig1, axs1 = plt.subplots(neofs//2, 2, figsize=(6, 5))
                if spectral:
                    fig2, axs2 = plt.subplots(neofs//2, 2, figsize=(6, 5))
            
                
                
                if log:
                    fig.suptitle(f'{OB}, EOF as Covariance: log10 APE density at {true_depth}m')
                    fig1.suptitle(f'{OB}, EOF PC: log10 APE density at {true_depth}m')
                    # fig2.suptitle(f'{OB}, EOF PC: log10 APE density at {true_depth}m, Power Spectral Density')
        
                else:
                    fig.suptitle(f'{OB}, EOF as Covariance: APE density at {true_depth}m')
                    fig1.suptitle(f'{OB}, EOF PC: APE density at {true_depth}m')
                    # fig2.suptitle(f'{OB}, EOF PC: APE density at {true_depth}m, Power Spectral Density')
        
                rolling_n = 12
                #creating dataframe
                df = pd.DataFrame()
                for e in range(neofs):
                    df[e] = pc1[:, e]
                rolling = df.rolling(rolling_n, center = True)
             
                for e in range(neofs):
                    basin, lonb, latb = eof1[e, :, :], lon_m, lat_m
                    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
                    axs[e//2, e%2].set_facecolor('darkgrey')
                    
                   
                    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
                    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
                    if OB in ['Arctic Ocean', 'Southern Ocean']:
                        basin, lonb, latb = crop_oceanbasin(eof1[e, :, :], lon, lat)
                        
                        
                        axs[e//2, e%2].coastlines()
                        basin_c, lon_c = add_cyclic_point(basin, coord=lon, axis=-1)
                        plot = axs[e//2, e%2].contourf(lon_c, latb, basin_c, #levels=clevs,
                                    cmap=plt.cm.RdBu_r, norm = norm, transform = ccrs.PlateCarree())
        
                        gl = axs[e//2, e%2].gridlines(draw_labels=True, xlocs=None, ylocs=None, color = 'black', alpha = 0.3)
                        gl.n_steps = 90
                        lat_lims = [latb[0], latb[-1]]
                        polarCentral_set_latlim(lat_lims, axs[e//2, e%2])
                    else:
                        axs[e//2, e%2].coastlines()
                        axs[e//2, e%2].gridlines(draw_labels = True, color = 'black', alpha = 0.3)
        
                        axs[e//2, e%2].set_extent(extent)
                        basin_c, lon_c = add_cyclic_point(basin, coord=lon, axis=-1)
                        plot = axs[e//2, e%2].contourf(lon_c, lat, basin_c, #levels=clevs,
                                    cmap=plt.cm.RdBu_r, norm = norm, transform = ccrs.PlateCarree())
            
                    if density:
                        label = 'APE Variance, $Jkg^{-1}$'
                    else:
                        label = 'APE Variance, $Jm^{-3}$'
                    fig.colorbar(plot, ax = axs[e//2, e%2], shrink=0.8, location = 'bottom', label = label) 
                    
                    axs1[e//2, e%2].plot(time, pc1[:, e], label = 'EOF')
                    axs1[e//2, e%2].plot(time, rolling.mean()[e], color = 'black',
                                         label = f'{rolling_n} month rolling mean')
                    axs1[e//2, e%2].set_xlabel('Year')
                    if OB == 'Arctic Ocean':
                        # ax.vlines(pd.date_range(f'1990-01-01', periods=1, freq='m'), 0, 4000, color = 'black', linestyle = ':', alpha = 0.7)
                        # ax.text(pd.date_range(f'2008-09-01', periods=1, freq='m'), 1000, '09-2016')
              
                        axs1[e//2, e%2].vlines(pd.date_range(f'2016-09-01', periods=1
                                                             , freq='m'),
                                               pc1[:, e].min(), pc1[:, e].max(),
                                               color = 'black', linestyle = ':',
                                               alpha = 0.7, label = f'2016-09')
        
                    if OB == 'Southern Ocean':
                        # ax.vlines(pd.date_range(f'1990-01-01', periods=1, freq='m'), 0, 4000, color = 'black', linestyle = ':', alpha = 0.7)
                        # ax.text(pd.date_range(f'2008-09-01', periods=1, freq='m'), 1000, '09-2016')
              
                        axs1[e//2, e%2].vlines(pd.date_range(f'2006-09-01', periods=1
                                                             , freq='m'),
                                               pc1[:, e].min(), pc1[:, e].max(),
                                               color = 'black', linestyle = ':',
                                               alpha = 0.7, label = f'2006-09')
        
                    axs[e//2, e%2].set_title(f'EOF {int(e+1)}: {round(fracs[e], 4)*100}%')
                    axs1[e//2, e%2].set_title(f'PC {int(e+1)}: {round(fracs[e], 4)*100}%')
                    if spectral:
                        axs2[e//2, e%2].set_title(f'PC {int(e+1)}: {round(fracs[e], 4)*100}%')
                        frequencies, psd_values = welch(pc1[:, e], 12)#, nperseg=100)
                        axs2[e//2, e%2].semilogy(frequencies, psd_values, label = 'Raw PC')
                        axs2[e//2, e%2].set_xlabel('Frequency, $year^{-1}$')
                        axs2[e//2, e%2].set_ylabel('Power, arb')
                        fig2.tight_layout()
                        if 'Pacific' in OB:
                            axs2[e//2, e%2].axvline(x = 0.5, label = '1/(2 years)', color = 'black', linestyle = ':')
                            axs2[e//2, e%2].axvline(x = 1/7, label = '1/(7 years)', color = 'black', linestyle = '--')
                        axs2[0, 0].legend()
        
            
                    
                fig.tight_layout()
                fig1.tight_layout()
                axs1[0, 0].legend()
                OB = OB.replace(" ", "") #removing space for filename purposes
                
            #     if log:
            #         fig.savefig(f'{plotdir}/EOF_log_onedepth_{OB}_{true_depth}_spatial{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
            #         fig1.savefig(f'{plotdir}/EOF_log_onedepth_{OB}_{true_depth}_PC{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
            #         if spectral:
            #             fig2.savefig(f'{plotdir}/EOF_log_onedepth_{OB}_{true_depth}_spectral{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
        
            #     else: 
            #         fig.savefig(f'{plotdir}/EOF_onedepth_{OB}_{true_depth}_spatial{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
            #         fig1.savefig(f'{plotdir}/EOF_onedepth_{OB}_{true_depth}_PC{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
            #         if spectral:
            #             fig2.savefig(f'{plotdir}/EOF_onedepth_{OB}_{true_depth}_spectral{SO_cutoff_lat}.pdf', bbox_inches = 'tight')
        
            #     plt.close(fig)
            #     plt.close(fig1)
            #     if spectral:
            #         plt.close(fig2)
            
            # # except:
        
        
            
            
