#!/usr/bin/env python

import pickle
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Forecast:
    pass

filename = 'review-forecast.png'

fig, ax = plt.subplots()
ax.set_yscale('log')

# ax.plot(self.goes_curve_t, self.goes_curve_y, 'b')
# 
# ax.plot(self.pred_curve_t, self.pred_curve_y, 'g')
# for i in range(24):
#     ax.plot(self.pred_max_t[i], self.pred_max_y[i], 'r')



proc = subprocess.Popen('ls archive/2015/10/05/*', shell = True, stdout=subprocess.PIPE)
for fn in proc.stdout:
    with(open(fn.strip(),'r')) as fp:
        f = pickle.load(fp)
        ax.plot(f.pred_max_t[23], f.pred_max_y[23])

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

        


plt.savefig(filename, dpi=200)
plt.close('all')
