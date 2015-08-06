#!/usr/bin/env python

import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
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

for d in range(1000):
    time_begin = datetime.datetime(2015,3,1) + datetime.timedelta(hours=d)
    time_end   = time_begin + datetime.timedelta(days=3)
    ret = session.query(GOES).filter(GOES.t>=time_begin).filter(GOES.t<=time_end).all()
    print time_begin, len(ret)
    x_data = []
    y_data = []
    for r in ret:
        x_data.append(r.t)
        y_data.append(r.xray_flux_long)



    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(x_data, y_data)

    days    = mdates.DayLocator()  # every day
    daysFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    hours   = mdates.HourLocator()
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(daysFmt)
    ax.xaxis.set_minor_locator(hours)

    ax.grid()
    fig.autofmt_xdate()


    plt.savefig('test2.png', dpi=300)

    plt.close('all')
    time.sleep(0.5)
