#!/usr/bin/env python
# -*- coding: utf-8 -*-
import astropy.time as time
import datetime
import pickle
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Forecast:
    pass

filename = 'review-forecast.png'

fig, ax = plt.subplots() # plt.subplots(figsize=mpl.figure.figaspect(0.3))
ax.set_yscale('log')


now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime


ts = [now, now-  datetime.timedelta(days=28)]
pats = ['archive/{:04}/{:02}/*/*'.format(t.year,t.month) for t in ts]
for pat in set(pats):
    print "loading " + pat
    proc = subprocess.Popen('ls ' + pat, shell = True, stdout=subprocess.PIPE)
    for fn in proc.stdout:
        with(open(fn.strip(),'r')) as fp:
            f = pickle.load(fp)
            ax.plot(f.pred_curve_t, f.pred_curve_y, color=(0,0.7,0), lw=0.1)
            ax.plot(f.pred_max_t[23], f.pred_max_y[23], 'r', lw=1)

ax.plot(f.goes_curve_t, f.goes_curve_y, color=(0,0,0.5), lw=1.5)
ax.plot(f.goes_curve_t, f.goes_curve_y, color=(0.2,0.2,1), lw=1)

days    = mdates.DayLocator()  # every day
daysFmt = mdates.DateFormatter('%Y-%m-%d')
hours   = mdates.HourLocator()
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(daysFmt)
ax.xaxis.set_minor_locator(hours)
ax.grid()
fig.autofmt_xdate()
ax.set_title('GOES Forecast at {}(TAI)'.format(now.strftime('%Y-%m-%d %H:%M:%S')))
ax.set_xlabel('International Atomic Time')
ax.set_ylabel(u'GOES Long[1-8Å] Xray Flux')
ax.set_xlim([now-datetime.timedelta(days=16), now+datetime.timedelta(days=1)])
ax.set_ylim([1e-7, 1e-4])        

plt.savefig(filename, dpi=200)
plt.close('all')
