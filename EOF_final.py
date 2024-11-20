# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:18:24 2024

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
import matplotlib.dates as mdates
import matplotlib.path as mpath
countries = gpd.read_file(datapath+'ne_110m_land.zip')

plotdir = f'EN4 Plots/EOF_Final'
startyear = 1960
endyear = 2023
nmonths = (startyear-endyear+1)*12
time =  pd.date_range(f'{startyear}-01-01', 
                             periods=nmonths, freq='m')


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
def EOF_ocean(APE, OB):
    OF = ocean_filters[OB]
    masksum = np.isnan(OF) + mask
    mask_ocean = (masksum>0)
    solver = make_EOFsolver(APE, mask_ocean)
    fracs = solver.varianceFraction(neofs)
    print(fracs)
    return solver, mask_ocean

def find_lonlatlims(eof):
    lon_where_1d = np.sum(eof[0, :, :]<np.inf, axis = 0)
    lonidx = np.where(lon_where_1d >0)[0]
    lat_where_1d = np.sum(eof[0, :, :]<np.inf, axis = 1)
    latidx = np.where(lat_where_1d >0)[0]
    latlims = [lat[latidx[0]], lat[latidx[-1]]]
    
    if True in ((lonidx[1:] - lonidx[0:-1]) > 100):
        idx_gap_1 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]]
        idx_gap_2 = lonidx[np.where((lonidx[1:] - lonidx[0:-1]) > 100)[0][0]+1]

        lonlims = [lon[idx_gap_2], lon[idx_gap_1]]

    else:
        lonlims = [lon[lonidx[0]], lon[lonidx[-1]]]
        
    return lonlims, latlims

def find_proj(OB):
    #Defining projections for spatial mode
    if OB == 'Arctic Ocean':
        proj = ccrs.NorthPolarStereo()
    elif OB == 'Southern Ocean':
        proj = ccrs.SouthPolarStereo()
    elif 'Pacific' in OB:
        proj = ccrs.PlateCarree(central_longitude = 180)
    else:
        proj = ccrs.PlateCarree()
    return proj

def plot_spatial_PC(APE, OB, n, figsize = (6.8, 3), cbarloc = 'right', flip = False):
    print(OB)
    solver, mask_ocean = EOF_ocean(APE, OB)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    eof = solver.eofsAsCovariance(neofs=neofs)
    pc = solver.pcs(npcs=neofs, pcscaling=1)
    if flip:
        eof = -eof
        pc = -pc
    
    df = pd.DataFrame()
    for e in range(neofs):
        df[e] = pc[:, e]
    rolling = df.rolling(12, center = True) #12 month rolling mean
    
    lonlims, latlims = find_lonlatlims(eof)
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
    fig = plt.figure(figsize = figsize, layout = 'tight')
    
    proj = find_proj(OB)
        
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        basin, lonb, latb = crop_oceanbasin(eof[n, :, :], lon, lat)
    else:
        basin, lonb, latb = eof[n, :, :], lon, lat

    ax_spatial = plt.subplot(121, projection = proj)
    ax_spatial.set_facecolor('darkgrey')
    ax_spatial.coastlines()
    basin_c, lon_c = add_cyclic_point(basin, coord=lonb, axis=-1)
    
    
    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    plot = ax_spatial.contourf(lon_c, latb, basin_c, 
                cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
    cbar = plt.colorbar(plot, ax = ax_spatial, location = cbarloc,
                 label = 'APE Variance, $Jkg^{-1}$')
    ax_spatial.set_ylim(latlims[0], latlims[-1])

    
    ax_PC = plt.subplot(122)
    ax_PC.plot(time, pc[:, n], color = 'royalblue')
    ax_PC.plot(time, rolling.mean()[n], color = 'black')
    ax_PC.set_ylabel(f'PC{n+1}, arb')
    ax_PC.set_xlabel('Time')
    ax_PC.xaxis.set_minor_locator(mdates.YearLocator(10))
    ax_PC.xaxis.set_major_locator(mdates.YearLocator(20))
    
    ticks = cbar.get_ticks()
    if cbarloc in ['bottom', 'top']:
        if 0 in ticks[::2]:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=False)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=True)
        else:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=True)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=False)
    
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        polarCentral_set_latlim(latlims, ax_spatial)
        gls = ax_spatial.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), lw=1, color="gray",
        y_inline=True, xlocs=range(-180,180,60), ylocs=range(0,90,15))
        for ea in gls.label_artists:
            pos = ea.get_position()
            if pos[0] != 180:
                print("Position:", pos, ea.get_text())
                ea.set_position([180, pos[1]])
    else:
        ax_spatial.set_extent(extent)
        ax_spatial.set_ylim(latlims[0], latlims[-1])
        gls = ax_spatial.gridlines(draw_labels = True, color = 'black', alpha = 0.2)
        gls.top_labels=False
        gls.right_labels=False 

        
    return fig, [ax_spatial, ax_PC]

def plot_spatial_PC_sep(APE, OB, n, figsize = (6, 3), cbarloc = 'right', flip = False):
    print(OB)
    proj = find_proj(OB)
    fig_sp, ax_sp = plt.subplots(figsize = (figsize), 
                                 subplot_kw = {'projection':proj})
    ax_sp.set_facecolor('darkgrey')
    fig_pc, ax_pc = plt.subplots(figsize = (figsize))
    
    solver, mask_ocean = EOF_ocean(APE, OB)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    eof = solver.eofsAsCovariance(neofs=neofs)
    pc = solver.pcs(npcs=neofs, pcscaling=1)
    if flip:
        eof = -eof
        pc = -pc
    
    df = pd.DataFrame()
    for e in range(neofs):
        df[e] = pc[:, e]
    rolling = df.rolling(12, center = True) #12 month rolling mean
    
    lonlims, latlims = find_lonlatlims(eof)
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
        
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        basin, lonb, latb = crop_oceanbasin(eof[n, :, :], lon, lat)
    else:
        basin, lonb, latb = eof[n, :, :], lon, lat

    ax_sp.coastlines()
    basin_c, lon_c = add_cyclic_point(basin, coord=lonb, axis=-1)
    
    
    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    plot = ax_sp.contourf(lon_c, latb, basin_c, 
                cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
    plt.colorbar(plot, ax = ax_sp, location = cbarloc,
                 label = 'APE Variance, $Jkg^{-1}$')
    ax_sp.set_ylim(latlims[0], latlims[-1])

    
    ax_pc.plot(time, pc[:, n], color = 'royalblue')
    ax_pc.plot(time, rolling.mean()[n], color = 'black')
    ax_pc.set_ylabel(f'PC{n+1}, arb')
    ax_pc.set_xlabel('Time')
    ax_pc.xaxis.set_minor_locator(mdates.YearLocator(10))
    ax_pc.xaxis.set_major_locator(mdates.YearLocator(20))

    
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        polarCentral_set_latlim(latlims, ax_sp)
        gls = ax_sp.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), lw=1, color="gray",
        y_inline=True, xlocs=range(-180,180,60), ylocs=range(0,90,15))
        for ea in gls.label_artists:
            pos = ea.get_position()
            if pos[0] != 180:
                print("Position:", pos, ea.get_text())
                ea.set_position([180, pos[1]])
    else:
        ax_sp.set_extent(extent)
        ax_sp.set_ylim(latlims[0], latlims[-1])
        gls = ax_sp.gridlines(draw_labels = True, color = 'black', alpha = 0.2)
        gls.top_labels=False
        gls.right_labels=False 

        
    return [fig_sp, fig_pc], [ax_sp, ax_pc]

def plot_multi_sep(APE, OB, ns, figsize = (6, 3), cbarloc = 'right',  vert = False):
    print(OB)
    proj = find_proj(OB)
    if vert:
        fig_sp, ax_sp = plt.subplots(len(ns), 1, figsize = (figsize), 
                                     subplot_kw = {'projection':proj},
                                     layout = 'constrained', sharex = True)
        fig_pc, ax_pc = plt.subplots(len(ns), 1, figsize = (6, 5),
                                     layout = 'constrained', sharex = True)
    else:
        fig_sp, ax_sp = plt.subplots(1, len(ns), figsize = (figsize), 
                                     subplot_kw = {'projection':proj},
                                     layout = 'constrained')
        fig_pc, ax_pc = plt.subplots(1, len(ns), figsize = (6, 2.3),
                                     layout = 'constrained')
    
    solver, mask_ocean = EOF_ocean(APE, OB)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    eof = solver.eofsAsCovariance(neofs=neofs)
    pc = solver.pcs(npcs=neofs, pcscaling=1)
    
    df = pd.DataFrame()
    for e in range(neofs):
        df[e] = pc[:, e]
    rolling = df.rolling(12, center = True) #12 month rolling mean
    
    lonlims, latlims = find_lonlatlims(eof)
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
    for i in range(len(ns)):    
        n = ns[i]
        if OB in ['Arctic Ocean', 'Southern Ocean']:
            basin, lonb, latb = crop_oceanbasin(eof[n, :, :], lon, lat)
        else:
            basin, lonb, latb = eof[n, :, :], lon, lat
            
        ax_sp[i].set_facecolor('darkgrey')
        ax_sp[i].coastlines()
        basin_c, lon_c = add_cyclic_point(basin, coord=lonb, axis=-1)
        
        
        maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
        norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
        
        plot = ax_sp[i].contourf(lon_c, latb, basin_c, 
                    cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
        cbar = plt.colorbar(plot, ax = ax_sp[i], location = cbarloc,
                     label = 'APE Variance, $Jkg^{-1}$', fraction = 0.8)
        ticks = cbar.get_ticks()
        if (cbarloc in ['bottom', 'top']):
            if 0 in ticks[::2]:
                cbar.ax.xaxis.set_ticks(ticks[::2], minor=False)
                cbar.ax.xaxis.set_ticks(ticks[1::2], minor=True)
            else:
                cbar.ax.xaxis.set_ticks(ticks[::2], minor=True)
                cbar.ax.xaxis.set_ticks(ticks[1::2], minor=False)
        if (cbarloc == 'right' and vert):
            if 0 in ticks[::2]:
                cbar.ax.yaxis.set_ticks(ticks[::2], minor=False)
                cbar.ax.yaxis.set_ticks(ticks[1::2], minor=True)
            else:
                cbar.ax.yaxis.set_ticks(ticks[::2], minor=True)
                cbar.ax.yaxis.set_ticks(ticks[1::2], minor=False)


        ax_sp[i].set_ylim(latlims[0], latlims[-1])

        
        ax_pc[i].plot(time, pc[:, n], color = 'royalblue')
        ax_pc[i].plot(time, rolling.mean()[n], color = 'black')
        ax_pc[i].set_ylabel(f'PC{n+1}, arb')
        ax_pc[i].set_xlabel('Time')
        ax_pc[i].xaxis.set_minor_locator(mdates.YearLocator(10))
        ax_pc[i].xaxis.set_major_locator(mdates.YearLocator(20))

    
        if OB in ['Arctic Ocean', 'Southern Ocean']:
            polarCentral_set_latlim(latlims, ax_sp[i])
            gls = ax_sp[i].gridlines(draw_labels=True, crs=ccrs.PlateCarree(), lw=1, color="gray",
            y_inline=True, xlocs=range(-180,180,60), ylocs=range(0,90,15))
            for ea in gls.label_artists:
                pos = ea.get_position()
                if pos[0] != 180:
                    print("Position:", pos, ea.get_text())
                    ea.set_position([180, pos[1]])
        else:
            ax_sp[i].set_extent(extent)
            ax_sp[i].set_ylim(latlims[0], latlims[-1])
            gls = ax_sp[i].gridlines(draw_labels = True, color = 'black', alpha = 0.2)
            gls.top_labels=False
            gls.right_labels=False 
            if vert and (i!=(len(ns)-1)):
                gls.bottom_labels = False

        
    return [fig_sp, fig_pc], [ax_sp, ax_pc]
    
def plot_spatial_spectral(APE, OB, n, figsize = (6.8, 3), cbarloc = 'right', flip = False):
    print(OB)
    solver, mask_ocean = EOF_ocean(APE, OB)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    eof = solver.eofsAsCovariance(neofs=neofs)
    pc = solver.pcs(npcs=neofs, pcscaling=1)
    if flip:
        eof = -eof
        pc = -pc
    
    df = pd.DataFrame()
    for e in range(neofs):
        df[e] = pc[:, e]
    rolling = df.rolling(12, center = True) #12 month rolling mean
    
    lonlims, latlims = find_lonlatlims(eof)
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
    fig = plt.figure(figsize = figsize, layout = 'tight')
    
    proj = find_proj(OB)
        
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        basin, lonb, latb = crop_oceanbasin(eof[n, :, :], lon, lat)
    else:
        basin, lonb, latb = eof[n, :, :], lon, lat

    ax_spatial = plt.subplot(121, projection = proj)
    ax_spatial.coastlines()
    ax_spatial.set_facecolor('darkgrey')
    basin_c, lon_c = add_cyclic_point(basin, coord=lonb, axis=-1)
    
    
    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    plot = ax_spatial.contourf(lon_c, latb, basin_c, 
                cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
    cbar = plt.colorbar(plot, ax = ax_spatial, location = cbarloc,
                        label = 'APE Variance, $Jkg^{-1}$')
    ticks = cbar.get_ticks()
    if cbarloc in ['bottom', 'top']:
        if 0 in ticks[::2]:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=False)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=True)
        else:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=True)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=False)
    
    
    frequencies, psd_values = welch(pc[:, n], 12)#, nperseg=100)
    
    
    ax_spec = plt.subplot(122)
    ax_spec.semilogy(frequencies, psd_values, label = 'Raw PC', color = 'navy')
    ax_spec.set_xlabel('Frequency, $year^{-1}$')
    ax_spec.set_ylabel('Power, arb')
    if 'Pacific' in OB:
        ax_spec.axvline(x = 0.5, label = '1/(2 years)', color = 'black', linestyle = ':')
        ax_spec.axvline(x = 1/7, label = '1/(7 years)', color = 'black', linestyle = '--')
        ax_spec.legend()
    
    
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        polarCentral_set_latlim(latlims, ax_spatial)
        gls = ax_spatial.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), lw=1, color="gray",
        y_inline=True, xlocs=range(-180,180,60), ylocs=range(0,90,15))
        for ea in gls.label_artists:
            pos = ea.get_position()
            if pos[0] != 180:
                print("Position:", pos, ea.get_text())
                ea.set_position([180, pos[1]])
    else:
        ax_spatial.set_extent(extent)
        ax_spatial.set_ylim(latlims[0], latlims[-1])
        gls = ax_spatial.gridlines(draw_labels = True, color = 'black', alpha = 0.2)
        gls.top_labels=False
        gls.right_labels=False 

    fig_pc, ax_pc = plt.subplots(figsize = figsize)
    ax_pc.plot(time, pc[:, n], color = 'royalblue')
    ax_pc.plot(time, rolling.mean()[n], color = 'black')
    ax_pc.set_ylabel(f'PC{n+1}, arb')
    ax_pc.set_xlabel('Time')
    ax_pc.xaxis.set_minor_locator(mdates.YearLocator(10))
    ax_pc.xaxis.set_major_locator(mdates.YearLocator(20))
    
    return [fig, fig_pc], [ax_spatial, ax_spec, ax_pc]

def plot_spatial_pc_spectral(APE, OB, n, figsize = (6.8, 3), cbarloc = 'right', flip = False):
    print(OB)
    solver, mask_ocean = EOF_ocean(APE, OB)
    
    lon_m = ma.masked_array(LON, mask = mask_ocean)
    lat_m = ma.masked_array(LAT, mask = mask_ocean)
    
    eof = solver.eofsAsCovariance(neofs=neofs)
    pc = solver.pcs(npcs=neofs, pcscaling=1)
    if flip:
        eof = -eof
        pc = -pc
    
    df = pd.DataFrame()
    for e in range(neofs):
        df[e] = pc[:, e]
    rolling = df.rolling(12, center = True) #12 month rolling mean
    
    lonlims, latlims = find_lonlatlims(eof)
    extent = [lonlims[0], lonlims[-1], latlims[0], latlims[-1]]
    
    fig = plt.figure(figsize = figsize, layout = 'tight')
    
    proj = find_proj(OB)
        
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        basin, lonb, latb = crop_oceanbasin(eof[n, :, :], lon, lat)
    else:
        basin, lonb, latb = eof[n, :, :], lon, lat

    ax_spatial = plt.subplot(131, projection = proj)
    ax_spatial.coastlines()
    ax_spatial.set_facecolor('darkgrey')
    basin_c, lon_c = add_cyclic_point(basin, coord=lonb, axis=-1)
    
    
    maxval = max(np.abs(np.nanmin(basin)), np.nanmax(basin))
    norm = TwoSlopeNorm(vcenter=0, vmin = -maxval, vmax = maxval)
    
    plot = ax_spatial.contourf(lon_c, latb, basin_c, 
                cmap='coolwarm', norm = norm, transform = ccrs.PlateCarree())
    cbar = plt.colorbar(plot, ax = ax_spatial, location = cbarloc,
                        label = 'APE Variance, $Jkg^{-1}$')
    ticks = cbar.get_ticks()
    if cbarloc in ['bottom', 'top']:
        if 0 in ticks[::2]:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=False)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=True)
        else:
            cbar.ax.xaxis.set_ticks(ticks[::2], minor=True)
            cbar.ax.xaxis.set_ticks(ticks[1::2], minor=False)
    
    ax_pc = plt.subplot(132)
    ax_pc.plot(time, pc[:, n], color = 'royalblue')
    ax_pc.plot(time, rolling.mean()[n], color = 'black')
    ax_pc.set_ylabel(f'PC{n+1}, arb')
    ax_pc.set_xlabel('Time')
    ax_pc.xaxis.set_minor_locator(mdates.YearLocator(10))
    ax_pc.xaxis.set_major_locator(mdates.YearLocator(20))
    
    
    
    frequencies, psd_values = welch(pc[:, n], 12)#, nperseg=100)
    
    
    ax_spec = plt.subplot(133)
    ax_spec.semilogy(frequencies, psd_values, label = 'Raw PC', color = 'navy')
    ax_spec.set_xlabel('Frequency, $year^{-1}$')
    ax_spec.set_ylabel('Power, arb')
    if 'Pacific' in OB:
        ax_spec.axvline(x = 0.5, label = '1/(2 years)', color = 'black', linestyle = ':')
        ax_spec.axvline(x = 1/7, label = '1/(7 years)', color = 'black', linestyle = '--')
        ax_spec.legend()
    
    
    if OB in ['Arctic Ocean', 'Southern Ocean']:
        polarCentral_set_latlim(latlims, ax_spatial)
        gls = ax_spatial.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), lw=1, color="gray",
        y_inline=True, xlocs=range(-180,180,60), ylocs=range(0,90,15))
        for ea in gls.label_artists:
            pos = ea.get_position()
            if pos[0] != 180:
                print("Position:", pos, ea.get_text())
                ea.set_position([180, pos[1]])
    else:
        ax_spatial.set_extent(extent)
        ax_spatial.set_ylim(latlims[0], latlims[-1])
        gls = ax_spatial.gridlines(draw_labels = True, color = 'black', alpha = 0.2)
        gls.top_labels=False
        gls.right_labels=False 


    return fig, [ax_spatial, ax_pc, ax_spec]

SO_cutoff_lat = -45

with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'rb') as f:
    ocean_filters = pickle.load(f)
with open(f'RegionFilters/ocean_filters-EN4_-45.pkl', 'rb') as f:
    IO45 = np.nan_to_num(pickle.load(f)['Indian Ocean'])
with open(f'RegionFilters/ocean_filters-EN4_-30.pkl', 'rb') as f:
    IO30 = pickle.load(f)['Indian Ocean']
IO_band = IO45-np.nan_to_num(IO30)
IO_band[np.where(IO_band == 0)] = np.nan
ocean_filters['IO Band'] = IO_band
ocean_filters['IO30'] = IO30

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
    
neofs = 4
#%%
if __name__ == '__main__':
    neofs = 4
    
    #setting up filters
    
    

    #%%
    depth = 10
    print(depth)
    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    #%%
    
    OB = 'Arctic Ocean'
    fig, axs = plot_spatial_PC(APE, OB, 0, cbarloc = 'left')
    x_dip = pd.date_range('2014-01-01', periods=1, freq='m')
    
    axs[1].axvline(x_dip, label = '2014-01', color = 'black', linestyle = ':')
    axs[1].legend(loc = 'lower left')
    
    
    fig.savefig(f'{plotdir}/AO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    
    OB = 'South Atlantic Ocean'
    figs, axs= plot_spatial_PC_sep(APE, OB, 0, figsize = (6, 2.7))
    axs[0].set_ylim(-45, 0)
    figs[0].savefig(f'{plotdir}/SAO_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/SAO_{true_depth}m_PC.pdf', bbox_inches = 'tight')
    #%%
    OB = 'Southern Ocean'
    figs, axs = plot_multi_sep(APE, OB, [0, 1], cbarloc = 'bottom', figsize = (6, 4))
    x_dip = pd.date_range('2006-09-01', periods=1, freq='m')
    axs[1][0].axvline(x_dip, label = '2006-09', color = 'black', linestyle = ':')
    axs[1][1].axvline(x_dip, label = '2006-09', color = 'black', linestyle = ':')
    axs[1][0].legend(loc = 'lower left')
    
    figs[0].savefig(f'{plotdir}/SO_{true_depth}m_SP_multi.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/SO_{true_depth}m_PC_multi.pdf', bbox_inches = 'tight')
    #%%
    depth = 100
    print(depth)
    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    
    fig, axs = plot_spatial_spectral(APE, 'IO30', 0, cbarloc='bottom')
    fig[0].savefig(f'{plotdir}/IO30_{true_depth}m_SPSP.pdf', bbox_inches = 'tight')
    fig[1].savefig(f'{plotdir}/IO30_{true_depth}m_PC.pdf', bbox_inches = 'tight')
    #%%
    depth = 150
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    #%%
    figs, axs = plot_spatial_spectral(APE, 'Indian Ocean', 0, cbarloc='bottom')
    figs[0].savefig(f'{plotdir}/IO_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/IO_{true_depth}m_PC.pdf', bbox_inches = 'tight')
    #%%
    figs, axs = plot_spatial_spectral(APE, 'South Pacific Ocean', 0, cbarloc='bottom', figsize = (6.8, 2.5), flip = True)
    figs[0].savefig(f'{plotdir}/SPO_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/SPO_{true_depth}m_spec.pdf', bbox_inches = 'tight')
    
    figs, axs = plot_spatial_spectral(APE, 'North Pacific Ocean', 0, cbarloc='bottom', figsize = (6.8, 2.5), flip = True)
    figs[0].savefig(f'{plotdir}/NPO_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/NPO_{true_depth}m_spec.pdf', bbox_inches = 'tight')
    
    figs, axs = plot_spatial_spectral(APE, 'Equatorial Pacific', 0, cbarloc='bottom', figsize = (6.8, 2.5), flip = True)
    figs[0].savefig(f'{plotdir}/EP_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/EP_{true_depth}m_spec.pdf', bbox_inches = 'tight')
    
    figs, axs = plot_multi_sep(APE, 'Equatorial Pacific', [1, 2, 3], cbarloc = 'right', figsize = (6, 4.5), vert = True)
    figs[0].savefig(f'{plotdir}/EP_{true_depth}m_SP_multi.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/EP_{true_depth}m_PC_multi.pdf', bbox_inches = 'tight')
    #%%
    depth = 200
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    
    figs, axs = plot_multi_sep(APE, 'North Atlantic Ocean', [0, 1], cbarloc = 'bottom', figsize = (6, 4))
    figs[0].savefig(f'{plotdir}/NAO_{true_depth}m_SP_multi.pdf', bbox_inches = 'tight')
    figs[1].savefig(f'{plotdir}/NAO_{true_depth}m_PC_multi.pdf', bbox_inches = 'tight')
    #%%
    depth = 400
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    
    fig, axs = plot_spatial_PC(APE, 'Southern Ocean', 0, cbarloc = 'left')
    x_dip = pd.date_range('2006-09-01', periods=1, freq='m')
    axs[1].axvline(x_dip, label = '2006-09', color = 'black', linestyle = ':')
    axs[1].legend(loc = 'lower left')
    fig.savefig(f'{plotdir}/SO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    
    fig, axs = plot_spatial_PC(APE, 'South Pacific Ocean', 0, flip = True, cbarloc = 'bottom', figsize = (6.8, 2.5))
    fig.savefig(f'{plotdir}/SPO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    fig, axs = plot_spatial_PC(APE, 'South Atlantic Ocean', 0, flip = True, cbarloc = 'bottom', figsize = (6.8, 3))
    fig.savefig(f'{plotdir}/SAO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    depth = 600
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    
    fig, axs = plot_spatial_PC(APE, 'South Pacific Ocean', 0, flip = True, cbarloc = 'bottom', figsize = (6.8, 2.5))
    fig.savefig(f'{plotdir}/SPO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    
    #%%
    depth = 800
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    #%%
    fig, axs = plot_spatial_PC(APE, 'Indian Ocean', 0, flip = False, cbarloc = 'bottom', figsize = (6.8, 3))
    fig.savefig(f'{plotdir}/IO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    fig, axs = plot_spatial_PC_sep(APE, 'IO Band', 0, cbarloc = 'bottom', figsize = (6.8, 2.2))
    fig[0].savefig(f'{plotdir}/IOBand_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    fig[1].savefig(f'{plotdir}/IOBand_{true_depth}m_PC.pdf', bbox_inches = 'tight')
    #%%
    fig, axs = plot_spatial_PC(APE, 'South Pacific Ocean', 0, flip = True, cbarloc = 'bottom', figsize = (6.8, 2.4))
    fig.savefig(f'{plotdir}/SPO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    fig, axs = plot_spatial_PC(APE, 'North Atlantic Ocean', 3, cbarloc = 'bottom', figsize = (6, 3))
    fig.savefig(f'{plotdir}/NAO_{true_depth}m_SP.pdf', bbox_inches = 'tight')
    # figs[1].savefig(f'{plotdir}/NAO_{true_depth}m_PC.pdf', bbox_inches = 'tight')
    
    #%%
    figs, axs = plot_spatial_PC(APE, 'South Atlantic Ocean', 0, flip = True, cbarloc = 'bottom', figsize = (6, 2.7))
    figs.savefig(f'{plotdir}/SAO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    fig, axs = plot_spatial_PC(APE, 'South Atlantic Ocean', 2, cbarloc = 'bottom', figsize = (6, 3))
    fig.savefig(f'{plotdir}/SAO_{true_depth}m_SP_3.pdf', bbox_inches = 'tight')
    # figs[1].savefig(f'{plotdir}/SAO_{true_depth}m_PC_3.pdf', bbox_inches = 'tight')
    
    #%%
    depth = 3000
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    
    fig, axs = plot_spatial_PC(APE, 'South Atlantic Ocean', 0, flip = False, cbarloc = 'bottom', figsize = (6.8, 3))
    fig.savefig(f'{plotdir}/SAO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    depth = 30
    print(depth)

    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    
    fig, axs = plot_spatial_PC(APE, 'IO30', 0, flip = False, cbarloc = 'bottom', figsize = (6.8, 3))
    fig.savefig(f'{plotdir}/SAO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    #%%
    depth = 1900
    print(depth)
 
    APE, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear, density = True)
    true_depth = int(np.round(true_depth[0], -1))
    print(true_depth)
    #%%
    n = 0
    fig, axs = plot_spatial_PC(APE, 'Indian Ocean', n, flip = False, cbarloc = 'bottom', figsize = (6.8, 3))
    fig.savefig(f'{plotdir}/IO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
    fig, axs = plot_spatial_PC(APE, 'South Pacific Ocean', n, flip = False, cbarloc = 'bottom', figsize = (6.8, 3))
    fig.savefig(f'{plotdir}/SPO_{true_depth}m_PC1.pdf', bbox_inches = 'tight')
