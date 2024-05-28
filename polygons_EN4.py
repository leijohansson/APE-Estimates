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


#import files with polygons for oceans/seas
worldseas = gpd.read_file('World_Seas_IHO_v3\World_Seas_IHO_v3.shp')

#read file to get lon lat bounds and resolution
datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')

#getting lon and lat
lat = data.lat.to_numpy()
lon = data.lon.to_numpy()
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


oceans = ['North Atlantic Ocean', 'South Atlantic Ocean',
          'North Pacific Ocean', 'South Pacific Ocean',
          'Indian Ocean', 'Southern Ocean', 'Arctic Ocean']
acros = oceans.copy()
def acronym(string):
    words = string.split()
    j = ''
    for w in words:
        j += w[0]
    return j
#creating ocean dataframe
oceans_df = worldseas[worldseas['NAME'].isin(oceans)]
#finding points in ocean polygons
pointInPolys= gpd.tools.sjoin(points, oceans_df, predicate="within", how='left')

#creating filters for different ocean basins
#colors for plotting
# colors = ['Reds_r', 'Oranges_r', 'plasma_r', 'Greens_r', 'Blues_r', 'Purples_r', 'Greys_r'] 
# c = 0 #color index for plotting
ocean_dict = {}
for name in oceans:
    LonLatBool = np.zeros(shape_2d)
    ocean_points = pointInPolys[pointInPolys.NAME == name]
    for pt in ocean_points.index:
        i = ocean_points.lon_i
        j = ocean_points.lat_j
        LonLatBool[j, i] = 1
        LonLatBool[LonLatBool == 0] = np.nan
    for j in range(len(lat)):
        if (LonLatBool[j, 179+1] == 1) and (LonLatBool[j, 179-1]==1):
            LonLatBool[j, 179] = 1
    ocean_dict[name] = LonLatBool.copy()
    # plotting to check that its correct
    # plt.imshow(LonLatBool, cmap = colors[c])
    
    #c += 1 #updating color index for plotting

#saving filters as a dictionary
with open('RegionFilters/ocean_filters-EN4.pkl', 'wb') as f:
    pickle.dump(ocean_dict, f)

#repeating above but for mediterranean and red sea
seas = ['Mediterranean Sea - Eastern Basin', 'Mediterranean Sea - Western Basin', 'Red Sea']
seas_df = worldseas[worldseas['NAME'].isin(seas)]
pointInPolys_sea= gpd.tools.sjoin(points, seas_df, predicate="within", how='left')

seas_dict = {}
for name in seas:
    LonLatBool = np.zeros(shape_2d)
    sea_points = pointInPolys[pointInPolys_sea.NAME == name]
    for pt in sea_points.index:
        i = sea_points.lon_i
        j = sea_points.lat_j
        LonLatBool[j, i] = 1
        LonLatBool[LonLatBool == 0] = np.nan
    seas_dict[name] = LonLatBool.copy()

with open('RegionFilters/sea_filters-EN4.pkl', 'wb') as f:
    pickle.dump(seas_dict, f)