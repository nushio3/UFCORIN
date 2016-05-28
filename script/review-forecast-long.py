#!/usr/bin/env python
# -*- coding: utf-8 -*-
import astropy.time as time
import datetime, os
import pickle
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

os.chdir(os.path.dirname(__file__))

def discrete_t(t):
    epoch = datetime.datetime(2011,1,1)
    dt = t - epoch
    return epoch + datetime.timedelta(seconds = int(dt.total_seconds()/720)*720)


class Forecast:
    pass

filename = 'review-forecast-long.png'
plt.rcParams['figure.figsize'] = (48.0,8.0)

fig, ax = plt.subplots() # plt.subplots(figsize=mpl.figure.figaspect(0.3))
ax.set_yscale('log')



now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime
t_begin = time.Time(datetime.datetime(2015,8,13),format='datetime',scale='utc').tai.datetime

t = now
ts = [now]
while True:
    t -=  datetime.timedelta(days=28)
    ts.append(t)
    if t < t_begin:
        break
ts.reverse()

pats = []
for t in ts:
    for d10 in range(4):
        pats.append('archive/{:04}/{:02}/{}*/*'.format(t.year,t.month,d10))
    
#pats = ['archive/2016/01/1?/*']
goes_curve_max = {}
f = None
for pat in pats:
    print "loading " + pat
    proc = subprocess.Popen('ls ' + pat, shell = True, stdout=subprocess.PIPE)
    for fn in proc.stdout:
        with(open(fn.strip(),'r')) as fp:
            try:
                f = pickle.load(fp)
                #ax.plot(f.pred_curve_t, f.pred_curve_y, color=(0,0.7,0), lw=0.1)
                ax.plot(f.pred_max_t[23][0], f.pred_max_y[23][0], 'mo', markersize=2.0, markeredgecolor='r', zorder = 300)
            except:
                continue

    if f is None :
        continue
    for i in range(len(f.goes_curve_t)):
        t = f.goes_curve_t[i]
        y = f.goes_curve_y[i]
        for j in range(-1,120):
            t2 = discrete_t(t - datetime.timedelta(seconds=j*720))
            try:
                y2 = goes_curve_max[t2]
                goes_curve_max[t2] = max(y2, y)
            except:
                goes_curve_max[t2] = y

    ax.plot(f.goes_curve_t, f.goes_curve_y, color=(0.66,0.66,1), lw=1.5, zorder = 200)
    ax.plot(f.goes_curve_t, f.goes_curve_y, color=(0,0,1), lw=1, zorder = 201)

gmdata = sorted(goes_curve_max.items())
ax.plot([kv[0] for kv in gmdata], [kv[1] for kv in gmdata], color=(1,0.75,0.75), lw=2, zorder=100)


months  = mdates.MonthLocator()  
days    = mdates.DayLocator()  # every day
daysFmt = mdates.DateFormatter('%Y-%m-%d')
hours   = mdates.HourLocator()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(daysFmt)
ax.xaxis.set_minor_locator(days)
ax.grid()
fig.autofmt_xdate()
ax.set_title('GOES Forecast till {}(TAI)'.format(now.strftime('%Y-%m-%d %H:%M:%S')))
ax.set_xlabel('International Atomic Time')
ax.set_ylabel(u'GOES Long[1-8Å] Xray Flux')
ax.set_xlim([t_begin, now+datetime.timedelta(days=1)])
ax.set_ylim([0.5e-7, 1e-3])        

plt.savefig(filename, dpi=100)
plt.close('all')

subprocess.call('cp review-forecast-long.png  ~/public_html',shell=True)
subprocess.call('scp review-forecast-long.png ufcorin@kyoto-76483it5.cloudapp.net:/var/www/html/wordpress/wp-content/uploads/2016/04/review-forecast-long.png', shell=True)
