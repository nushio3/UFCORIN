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

# ax.plot(self.goes_curve_t, self.goes_curve_y, 'b')
# 
# ax.plot(self.pred_curve_t, self.pred_curve_y, 'g')
# for i in range(24):
#     ax.plot(self.pred_max_t[i], self.pred_max_y[i], 'r')

# now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime
now = datetime.date(2015,10,1)

# for pat in ['archive/2015/09/2?/??00*', 'archive/2015/09/3?/??00*', 'archive/2015/10/*/??00*']:
#     proc = subprocess.Popen('ls ' + pat, shell = True, stdout=subprocess.PIPE)
#     for fn in proc.stdout:
#         with(open(fn.strip(),'r')) as fp:
#             f = pickle.load(fp)
#             ax.plot(f.pred_max_t[23], f.pred_max_y[23], 'r')
# 
# for pat in ['archive/2015/09/2?/??00*', 'archive/2015/09/3?/??00*', 'archive/2015/10/*/??00*']:
#     proc = subprocess.Popen('ls ' + pat, shell = True, stdout=subprocess.PIPE)
#     for fn in proc.stdout:
#         with(open(fn.strip(),'r')) as fp:
#             f = pickle.load(fp)
#             ax.plot(f.pred_curve_t, f.pred_curve_y, 'g')

for pat in ['archive/2015/10/01/0000*']:
    proc = subprocess.Popen('ls ' + pat, shell = True, stdout=subprocess.PIPE)
    for fn in proc.stdout:
        with(open(fn.strip(),'r')) as fp:
            f = pickle.load(fp)
            ax.plot(f.pred_curve_t, f.pred_curve_y, 'g', lw=3)
            ax.plot(f.pred_max_t[23], f.pred_max_y[23], 'r', lw=3)

ax.plot(f.goes_curve_t, f.goes_curve_y, 'b', lw=3)

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
ax.set_xlim([datetime.date(2015,9,28), datetime.date(2015,10,2)])
ax.set_ylim([5e-7, 1e-4])        

plt.savefig(filename, dpi=200)
plt.close('all')
