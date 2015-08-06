#!/usr/bin/env python

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import random
import re
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import time
import urllib2

with open('.mysqlpass','r') as fp:
    password = fp.read().strip()

Base = declarative_base()

class GOES(Base):
    __tablename__ = 'goes_xray_flux'

    t = sql.Column(sql.DateTime, primary_key=True)
    xray_flux_long = sql.Column(sql.Float)
    xray_flux_short = sql.Column(sql.Float)


engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))

Session = sessionmaker(bind=engine)
session = Session()

global lightcurve, time_begin, time_end
lightcurve = dict()

# todo: optimize this by use of sortedcontainers
def goes_future_max(t0, dt):
    t = t0
    ret = 0
    while t<=t0+dt:
        t += datetime.timedelta(minutes=1)
        if lightcurve.has_key(t):
            ret=max(ret, lightcurve[t])
    return ret

while True:
    # time_begin = datetime.datetime(2014,3,8) + datetime.timedelta(hours=d)
    d = random.randrange(365*5*24)
    time_begin = datetime.datetime(2011,1,1) +  datetime.timedelta(hours=d)
    window_days = 10
    time_end   = time_begin + datetime.timedelta(days=window_days)
    ret = session.query(GOES).filter(GOES.t>=time_begin).filter(GOES.t<=time_end).all()
    print time_begin, len(ret)
    if  len(ret) < 0.95*24*60*window_days : continue

    t=time_begin
    while t <= time_end:
        lightcurve[t]=-0.0
        t+=datetime.timedelta(minutes=1)

    for r in ret:
        lightcurve[r.t] = r.xray_flux_long

    t_data = []
    goes_flux = []
    t=time_begin
    while t <= time_end:
        t_data.append(t)
        goes_flux.append(lightcurve[t])
        t+=datetime.timedelta(minutes=1)

    goes_max = []
    for t in t_data:
        goes_max.append(goes_future_max(t,datetime.timedelta(hours=4)))

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(t_data, goes_flux,'b')
    ax.plot(t_data, goes_max,'g')

    days    = mdates.DayLocator()  # every day
    daysFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    hours   = mdates.HourLocator()
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(daysFmt)
    ax.xaxis.set_minor_locator(hours)

    ax.grid()
    fig.autofmt_xdate()


    plt.savefig('test2.png', dpi=200)

    plt.close('all')
    time.sleep(0.5)
