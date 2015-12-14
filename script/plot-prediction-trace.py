#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlalchemy as sql
from   sqlalchemy.orm import sessionmaker
import datetime, pickle, sys, os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import goes.schema as goes


def visualize(gt, gx, curves, filename):
    fig, ax = plt.subplots()
    ax.set_yscale('log')

    ax.plot(gt,gx, 'r')
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
    

# Obtain MySQL Password
with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()
engine = sql.create_engine('mysql+pymysql://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))

Session = sessionmaker(bind=engine)
session = Session()
GOES = goes.GOES


dat = pickle.load(open(sys.argv[1]))

dat=dat[-5*24*10:-1]
#dat=dat[0:5*24*10]

curves = []
ct = []; cp = []; co = []
for l in dat:
    t,p,o = l
    if len(ct)>0 and t-ct[-1] > datetime.timedelta(seconds=800):
        curves.append((ct,cp,co))
        ct = []; cp = []; co = []
    else:
        ct.append(t); cp.append(p); co.append(o)

curves.append((ct,cp,co))

time_begin=dat[0][0]
time_end=dat[-1][0]
ret_goes = session.query(GOES).filter(GOES.t_tai>=time_begin, GOES.t_tai<=time_end).all()

gt = []; gx = []
for row in ret_goes:
    gt.append(row.t_tai)
    gx.append(row.xray_flux_long)


visualize(gt, gx, curves,'test.png')
