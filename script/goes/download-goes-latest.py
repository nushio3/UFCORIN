#!/usr/bin/env python

import datetime
import re
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
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
try:
    GOES.metadata.create_all(engine)
except:
    None
    # the table already exists

Session = sessionmaker(bind=engine)
session = Session()

fp = urllib2.urlopen('http://services.swpc.noaa.gov/text/goes-xray-flux-primary.txt')
con = fp.read()
lines = con.split('\n')
for l in lines:
    match = re.search('^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',l)
    if not match : continue
    year=int(match.group(1))
    month=int(match.group(2))
    day=int(match.group(3))
    hour=int(match.group(4)[0:2])
    minute=int(match.group(4)[2:4])
    flux_short=float(match.group(7))
    flux_long=float(match.group(8))
    
    goes = GOES(t=datetime.datetime(year,month,day,hour,minute,0), xray_flux_long=flux_long, xray_flux_short=flux_short)
    session.merge(goes)
session.commit()
