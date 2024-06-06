# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:18:13 2024

@author: Linne
"""

from EN4_singledepth import EN4_singledepth_time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation

startyear = 1960
endyear = 2020
depth = 400
nlevs = 15
vmin = -3

cont = False

data, time, lon, lat, true_depth = EN4_singledepth_time(depth, startyear, endyear)
true_depth = int(np.round(true_depth[0]))
extent = [lon[0], lon[-1], lat[0], lat[-1]]
log10 = np.log10(data)

fig, ax = plt.subplots(figsize = (18, 12))
fig.suptitle(f'APE density at {true_depth}m', y = 0.94)
Z = log10[0, :, :]

imshow = ax.imshow(np.flip(Z, axis = 0), vmin = vmin, vmax = np.nanmax(log10), extent = extent)
plt.colorbar(imshow, location = 'bottom', ax = ax, label = 'APE density, $Jm^{-3}$')
# if cont:
#     contour = ax.contour(lon, lat, Z, colors = 'black', levels = nlevs, vmin = vmin)
ax.set_title(f'{time[0].strftime("%Y-%m")}')

def update(frame):
    Z = log10[frame, :, :]
    imshow.set_data(Z)
    ax.set_title(f'{time[frame].strftime("%Y-%m")}')
    # if cont:
    #     for c in contour.collections:
    #         c.remove()
    #     contour = ax.contour(lon, lat, Z, colors = 'black', levels = nlevs)
    #     return imshow, contour.collections
    # else:
    return [imshow]
    
ani = animation.FuncAnimation(fig, update, frames = np.arange(1, len(time)), blit = True)
ani.save(f'EN4_{true_depth}.gif', writer = 'Pillow')
