#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime, pickle, sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def visualize(curves, filename):
    fig, ax = plt.subplots()
    ax.set_yscale('log')

    for curve_t, curve_p, curve_o in curves:
        ax.plot(curve_t, curve_o, 'g')
        ax.plot(curve_t, curve_p, 'b')

    days    = mdates.YearLocator()  # every year
    daysFmt = mdates.DateFormatter('%Y-%m-%d')
    hours   = mdates.MonthLocator()
#    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(daysFmt)
#    ax.xaxis.set_minor_locator(hours)
    ax.grid()
    fig.autofmt_xdate()
    ax.set_title('GOES Forecast for 2011-2014')
    ax.set_xlabel('International Atomic Time')
    ax.set_ylabel(u'GOES Long[1-8Å] Xray Flux')

    plt.savefig(filename, dpi=200)
    plt.close('all')
    


dat = pickle.load(open(sys.argv[1]))

dat=dat[-5*24*10:-1]

curves = []
ct = []; cp = []; co = []
for l in dat:
    t,p,o = l
    if len(ct)>0 and t-ct[0] > datetime.timedelta(800):
        curves.append((ct,cp,co))
        ct = []; cp = []; co = []
    else:
        ct.append(t); cp.append(p); co.append(o)

curves.append((ct,cp,co))

visualize(curves,'test.png')
