# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:20:34 2023

@author: Linne

Original creation of filter for IO and PO with no polygons
Coordinates provided by Remi Tailleux
"""

import geopandas as gpd
from shapely.geometry import Polygon, Point
import shapely.plotting
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

io_long=[100, 100, 55, 22, 22, 146, 146, 133.9, 126.94, 123.62, 120.92,
         117.42, 114.11, 107.79, 102.57, 102.57, 98.79, 100]
                     
io_lat=[20,40,40,20,-90,-90, -41, -12.48, -8.58, -8.39, -8.7, -8.82,
        -8.02, -7.04, -3.784 , 2.9, 10, 20]

io_polygon= Polygon(zip(io_long, io_lat))
gs_io = gpd.GeoSeries([io_polygon])

# io_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])

po_long = [100, 140, 240, 260, 272.59, 276.5, 278.65, 280.73, 295.217 ,
           290 , 300, 294, 290, 146, 146, 133.9, 126.94, 123.62, 120.92,
           117.42, 114.11,107.79, 102.57, 102.57, 98.79, 100]
     
po_lat = [20, 66, 66, 19.55, 13.97, 9.6, 8.1, 9.33, 0, -52, -64.5,
          -67.5, -90, -90, -41,-12.48, -8.58, -8.39, -8.7, -8.82, -8.02,
          -7.04, -3.784 , 2.9, 10, 20]
po_polygon = Polygon(zip(po_long, po_lat))
gs_po = gpd.GeoSeries([po_polygon])
# po_polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])

countries = gpd.read_file('ne_110m_land.zip')
fig, ax = plt.subplots()
# plt.figure(2)
countries.plot(color="lightgrey",ax = ax)
gs_io.plot(color = 'red',ax = ax, alpha = 0.3)
gs_po.plot(color = 'blue',ax = ax, alpha = 0.3)




datadir = 'Data' 
filename = 'EN.4.2.2.f.analysis.g10.195001.nc'
data = xr.open_dataset(f'{datadir}/{filename}')
lon = data.lon.to_numpy()
lat = data.lat.to_numpy()
lon_all = np.zeros(len(lon)*len(lat))
lat_all = np.zeros(len(lon)*len(lat))
lon_idx = lon_all.copy()
lat_idx = lat_all.copy()

for i in range(len(lat)-1):
    lon_all[i*len(lon):(i+1)*len(lon)] = lon
    lon_idx[i*len(lon):(i+1)*len(lon)] = np.arange(len(lon))
    lat_all[i*len(lon):(i+1)*len(lon)] = lat[i]
    lat_idx[i*len(lon):(i+1)*len(lon)] = i
    

df = pd.DataFrame({'lon':lon_all, 'lat':lat_all, 'lat_i':lat_idx,
                    'lon_i':lon_idx})
df['coords'] = list(zip(df['lon'],df['lat']))
df['coords'] = df['coords'].apply(Point)
points = gpd.GeoDataFrame(df, geometry='coords')
io_df = gpd.GeoDataFrame(['IO'], geometry = [io_polygon])
po_df = gpd.GeoDataFrame(['PO'], geometry = [po_polygon])
pointInPolys_io = gpd.tools.sjoin(points, io_df, predicate="within", how='left')
pnt_IO = points[pointInPolys_io[0]=='IO']
pointInPolys_po = gpd.tools.sjoin(points, po_df, predicate="within", how='left')
pnt_PO = points[pointInPolys_po[0]=='PO']

T = data.temperature.to_numpy().squeeze()
LatLonBool_IO = np.zeros(T.shape[1:])
for i in pnt_IO.index:
    x, y = pnt_IO.loc[i][['lat_i', 'lon_i']]
    LatLonBool_IO[int(x), int(y)] = 1
D3_BoolIO = np.zeros(T.shape)
for d in range(T.shape[0]):
    D3_BoolIO[d, :, :] = LatLonBool_IO
np.save('RegionFilters/IO_filter', D3_BoolIO)

LatLonBool_PO = np.zeros(T.shape[1:])
for i in pnt_PO.index:
    x, y = pnt_PO.loc[i][['lat_i', 'lon_i']]
    LatLonBool_PO[int(x), int(y)] = 1
D3_BoolPO = np.zeros(T.shape)
for d in range(T.shape[0]):
    D3_BoolPO[d, :, :] = LatLonBool_PO
np.save('RegionFilters/PO_filter', D3_BoolPO)


