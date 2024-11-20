# -*- coding: utf-8 -*-
"""
Created on Thu May  2 23:12:16 2024

@author: Linne

Creating ocean filters using polygons for EN4 data
"""

import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import pickle 
from FuncsAPE import datapath
def acronym(string):
    words = string.split()
    j = ''
    for w in words:
        j += w[0]
    return j

SO_cutoff_lat = -45

oceans = ['North Atlantic Ocean', 'South Atlantic Ocean',
          'North Pacific Ocean', 'South Pacific Ocean',
          'Indian Ocean', 'Southern Ocean', 'Arctic Ocean',
          'Mediterranean Region', 'Baltic Sea',
          'South China and Easter Archipelagic Seas']

if __name__ == '__main__':
    #import files with polygons for oceans/seas
    worldseas = gpd.read_file(datapath + 'goas\goas_v01.shp')
    
    #read file to get lon lat bounds and resolution
    datadir = datapath + 'Data' 
    filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
    data = xr.open_dataset(f'{datadir}/{filename}')
    
    #getting lon and lat
    lat = data.lat.to_numpy()
    lon = data.lon.to_numpy()
    
    extent = [lon[0], lon[-1], lat[0], lat[-1]]
    #converting lon to same range (-180 to 180)
    oor_lon = np.where(lon > 180)[0]
    lon[oor_lon] = lon[oor_lon] - 360
    
    shape_2d = data.temperature.to_numpy()[0, 0, :, :].shape
    
    #meshgrid of lon lat
    LON, LAT = np.meshgrid(lon, lat)
    lon_I, lat_J = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))
    
    #creating dataframe of grid points
    values = {'lon': LON.flatten(), 'lat': LAT.flatten(), 'lon_i': lon_I.flatten(),
              'lat_j': lat_J.flatten()}
    df = pd.DataFrame(values)
    df['coords'] = list(zip(df['lon'],df['lat']))
    df['coords'] = df['coords'].apply(Point)
    points = gpd.GeoDataFrame(df, geometry='coords', crs = 'epsg:4326')
    

    acros = oceans.copy()
    #creating ocean dataframe
    # oceans_df = worldseas[worldseas['NAME'].isin(oceans)]
    oceans_df = worldseas[worldseas['name'].isin(oceans)]
    
    #finding points in ocean polygons
    pointInPolys= gpd.tools.sjoin(points, oceans_df, predicate="within", how='left')
    
    #creating filters for different ocean basins
    #colors for plotting
    colors = ['Reds_r', 'Oranges_r', 'plasma_r', 'Greens_r', 'Blues_r', 'Purples_r', 'Greys_r', 'summer', 'spring', 'autumn'] 
    c = 0 #color index for plotting
    ocean_dict = {}
    plotting = np.zeros(shape_2d)
    plt.figure()
    for name in oceans:
        LonLatBool = np.zeros(shape_2d)
        # ocean_points = pointInPolys[pointInPolys.NAME == name]
        ocean_points = pointInPolys[pointInPolys.name == name]
        for pt in ocean_points.index:
            i = ocean_points.lon_i
            j = ocean_points.lat_j
            LonLatBool[j, i] = 1
        if name == 'Southern Ocean':
            LonLatBool[np.where(LAT<SO_cutoff_lat)] = 1
        else:
            LonLatBool[np.where(LAT<SO_cutoff_lat)] = 0
        if name == 'Indian Ocean':
            print(colors[c])
            sum_bool = (LAT >= 13).astype(int) + (LON < 44).astype(int)
            LonLatBool[np.where(sum_bool >1)] = 0
       
        plotting += LonLatBool*(c+1)
        LonLatBool[LonLatBool == 0] = np.nan
        for j in range(len(lat)):
            if (LonLatBool[j, 179+1] == 1) and (LonLatBool[j, 179-1]==1):
                LonLatBool[j, 179] = 1
            if (LonLatBool[j, 0] == 1) and (LonLatBool[j, -2]==1):
                LonLatBool[j, -1] = 1
        ocean_dict[name] = LonLatBool.copy()
        
        # plotting to check that its correct
        plt.imshow(np.flip(LonLatBool,  axis = 0), cmap = colors[c],
                    extent = extent)
        
        c += 1 #updating color index for plotting
    pacific = np.nan_to_num(ocean_dict['South Pacific Ocean']) +\
        np.nan_to_num(ocean_dict['North Pacific Ocean'])
    eq_pacific = np.zeros(pacific.shape)
    eq_pacific[np.where(np.abs(LAT) <= 20)] = 1
    
    eq_pacific = pacific * eq_pacific
    eq_pacific[eq_pacific == 0] = np.nan
    plt.imshow(np.flip(eq_pacific,  axis = 0), cmap = 'cividis',
                extent = extent)
    ocean_dict['Equatorial Pacific'] = eq_pacific
    #saving filters as a dictionary
    with open(f'RegionFilters/ocean_filters-EN4_{SO_cutoff_lat}.pkl', 'wb') as f:
        pickle.dump(ocean_dict, f)
    # plt.contour(lon, lat, plotting)
# 