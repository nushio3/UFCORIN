#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import math as M

for d in range(10):
    n=1024/(2**d)
    location_x = np.float32(np.array(n*[n*[0]]))
    location_y = np.float32(np.array(n*[n*[0]]))
    ox = n/2-0.5
    oy = n/2-0.5
    r0 = n*450.0/1024.0

    for iy in range(n):
        for ix in range(n):
            x = (ix - ox) / r0
            y = (iy - oy) / r0
            r = M.sqrt(x**2 + y**2)
            if r < 1:
                location_x[iy][ix]=M.asin(x/(M.cos(M.asin(y))))
                location_y[iy][ix]=M.asin(y)
            else:
                location_x[iy][ix]=-4
                location_y[iy][ix]=0
#gnuplot> splot asin(y)
#gnuplot> splot


    dpi=200
    plt.figure(figsize=(8,6),dpi=dpi)
    fig, ax = plt.subplots()
    circle1=plt.Circle((n/2,n/2),r0,edgecolor='black',fill=False)
    cmap = plt.get_cmap('bwr')
    cax = ax.imshow(location_x,cmap=cmap,extent=(0,n,0,n),vmin=-2.0,vmax=2.0)
    cbar=fig.colorbar(cax)
    fig.gca().add_artist(circle1)
    fig.savefig('test_{}_x.png'.format(n),dpi=dpi)

    cax = ax.imshow(location_y,cmap=cmap,extent=(0,n,0,n),vmin=-2.0,vmax=2.0)
    fig.gca().add_artist(circle1)
    fig.savefig('test_{}_y.png'.format(n),dpi=dpi)
    plt.close('all')
