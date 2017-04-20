#!/usr/bin/env python3
import astropy.time as time
import datetime, os,math,sys
import pickle
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Forecast:
    pass

for fn in sys.argv:
    with open(fn,"rb") as fp:
        forecast = pickle.load(fp)
    print(dir(forecast))
