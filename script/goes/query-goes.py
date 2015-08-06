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

global lightcurve
lightcurve = dict()

while True:
    # time_begin = datetime.datetime(2014,3,8) + datetime.timedelta(hours=d)
    d = random.randrange(365*5*24)
    time_begin = datetime.datetime(2011,1,1) +  datetime.timedelta(hours=d)
    window_days = 10
    time_end   = time_begin + datetime.timedelta(days=window_days)
    ret = session.query(GOES).filter(GOES.t>=time_begin).filter(GOES.t<=time_end).all()
    print time_begin, len(ret)
    if  len(ret) < 0.95*24*60*window_days : continue
    t_data = []
    goes_flux = []
    for r in ret:
        t_data.append(r.t)
        goes_flux.append(r.xray_flux_long)
        lightcurve[r.t] = r.xray_flux_long


    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(t_data, goes_flux)

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
