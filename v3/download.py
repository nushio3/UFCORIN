#!/usr/bin/env python
import datetime, os, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
from astropy.io import fits
from astropy import units as u

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import sunpy.map

import observational_data

img = get_aia_image(193, datetime.datetime(2011,1,1,0,0))

img.plot()
plt.colorbar()
plt.savefig('test.png')
