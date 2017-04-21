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

os.chdir(os.path.dirname(__file__))

def discrete_t(t):
    dt = t - datetime.datetime(2011,1,1)
    return int(dt.total_seconds()/720)


class Forecast:
    def generate_ccmc_submission(self):
        now = datetime.datetime.now()
        filename = "UFCORIN_1_{}.txt".format(now.strftime("%Y%m%d_%H%M"))

        ys_log = [math.log10(y[0]) for y in self.pred_max_y]
        pred_mean = ys_log[23]
        pred_stddev = max([abs(ys_log[i] - pred_mean) for i in [19,20,21,22]])
        def cdf(y):
            # cumulative gaussian distribution
            return 0.5 * (1 + math.erf((y-pred_mean)/(math.sqrt(2.0) * pred_stddev)))
            # x = (y-pred_mean) / pred_stddev
            # if x > 0:
            #     return 0.5*(2-math.exp(-x))
            # else:
            #     return 0.5*math.exp(x)
        # def cdf(y):
        #     # cumulative gaussian distribution
        #     # return 0.5 * (1 + math.erf((y-pred_mean)/(math.sqrt(2.0) * pred_stddev)))
        #     return (1-0.5)**((10.0**y / 10.0**pred_mean) ** (-0.53))
        self.prob_x = 1 - cdf(-4)
        self.prob_m = 1 - cdf(-5)
        self.prob_c = 1 - cdf(-6)
        self.time_begin = self.pred_max_t[23][0]
        self.time_end   = self.pred_max_t[23][1]

filename = 'review-forecast-long.png'
plt.rcParams['figure.figsize'] = (48.0,8.0)

fig, ax = plt.subplots() # plt.subplots(figsize=mpl.figure.figaspect(0.3))
ax.set_yscale('log')


now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime

t = now
ts = [now]
while t > datetime.datetime(2015,8,1):
    t -=  datetime.timedelta(days=28)
    ts.append(t)
ts.reverse()


pats = []
for t in ts:
    for d10 in range(4):
        pats.append('archive/{:04}/{:02}/{}*/*'.format(t.year,t.month,d10))

if 'debug' in sys.argv:
    pats = ['archive/2017/03/0?/*']

goes_curve_max = {}
goes_curve_pred = {}
goes_curve_obs = {}
f = None

fp_prob_log = open("probability_log.txt","w")
fp_obs_log = open("observation_log.txt","w")

for pat in pats:
    print "loading " + pat
    proc = subprocess.Popen('ls ' + pat, shell = True, stdout=subprocess.PIPE)
    for fn in proc.stdout:
        with(open(fn.strip(),'r')) as fp:
            try:
                f = pickle.load(fp)
                ax.plot(f.pred_curve_t, f.pred_curve_y, color=(0,0.7,0), lw=0.1)

                pred_t = f.pred_max_t[23][0]
                pred_y = f.pred_max_y[23][0]
                ax.plot(pred_t, pred_y, 'mo', markersize=2.0, markeredgecolor='r')
                goes_curve_pred[discrete_t(pred_t)] = pred_y

                f.generate_ccmc_submission()
                msg = "{} {} {} {}\n".format(discrete_t(pred_t), f.prob_x, f.prob_m, f.prob_c)
                fp_prob_log.write(msg)

            except:
                print "unpickleable", fn
                continue

    if f is None :
        continue
    for i in range(len(f.goes_curve_t)):
        t = f.goes_curve_t[i]
        y = f.goes_curve_y[i]
        for j in range(-1,120):
            t2 = t - datetime.timedelta(seconds=j*720)
            try:
                y2 = goes_curve_max[t2]
                goes_curve_max[t2] = max(y2, y)
            except:
                goes_curve_max[t2] = y
            goes_curve_obs[discrete_t(t2)] = goes_curve_max[t2]

    for d_t in sorted(goes_curve_obs.keys()):
        msg = "{} {}\n".format(d_t,  goes_curve_max[t2])
        fp_obs_log.write(msg)

    ax.plot(f.goes_curve_t, f.goes_curve_y, color=(0,0,0.5), lw=1.5)
    ax.plot(f.goes_curve_t, f.goes_curve_y, color=(0.2,0.2,1), lw=1)


# construct contingency table

contingency_table = {}
for clas in range(-7,-3):
    contingency_table[clas] = {}
    for p in [False,True]:
        for o in [False,True] :
            contingency_table[clas][(p,o)] = 0


for t,y_pred in goes_curve_pred.iteritems():
    if not t in goes_curve_obs:
        continue
    try:
        y_obs = goes_curve_obs[t]
        clas_obs  = int(math.floor(math.log10(float(y_obs ))))
        clas_pred = int(math.floor(math.log10(float(y_pred))))

        for clas in range(-7,-3):
            flag_obs = clas_obs >= clas
            flag_pred= clas_pred>= clas
            contingency_table[clas][(flag_pred,flag_obs)] += 1
    except:
        continue


for c,tbl in contingency_table.iteritems():
    print c
    print tbl
    tp = float(tbl[(True,True)])
    fp = float(tbl[(True,False)])
    fn = float(tbl[(False,True)])
    tn = float(tbl[(False, False)])

    tss = tp/(tp+fn+1e-30) - fp/(fp+tn+1e-30)
    print tss
