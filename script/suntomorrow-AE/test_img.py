#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

fig, ax = plt.subplots()


circle1=plt.Circle((512,512),450,edgecolor='black',fill=False)

img = np.load('copy-work/0000.npz')['img']
cmap = plt.get_cmap('bwr')
cax = ax.imshow(img,cmap=cmap,extent=(0,1024,0,1024),vmin=-100,vmax=100)
cbar=fig.colorbar(cax)
fig.gca().add_artist(circle1)
ax.set_title('Solar Image')
fig.savefig('/home/ubuntu/public_html/out.png')

