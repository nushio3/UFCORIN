#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

for fn in sys.argv[1:]:
    with open(fn,"r") as fp:
        forecast = pickle.load(fp)
    print(dir(forecast))
