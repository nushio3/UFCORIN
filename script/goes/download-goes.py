#!/usr/bin/env python

import astropy.time as time
import calendar
import datetime
import re
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import urllib2

with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()

Base = declarative_base()

class GOES(Base):
    __tablename__ = 'goes_xray_flux'

    t_tai = sql.Column(sql.DateTime, primary_key=True)
    xray_flux_long = sql.Column(sql.Float)
    xray_flux_short = sql.Column(sql.Float)





engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))
try:
    GOES.metadata.create_all(engine)
except:
    pass # the table already exists

Session = sessionmaker(bind=engine)
session = Session()

def write_db(con):
    data_part=False
    ctr = 0
    for l in con.split('\n'):
        if l.strip() == 'data:' :
            data_part=True
        if data_part:
            words = l.split(',')
            if len(words) < 7 : continue
            match = re.search('^(\d+)-(\d+)-(\d+)\s+(\d+):(\d+)',words[0])
            if not match :
                continue

            a_qf = int(words[1])
            b_qf = int(words[4])
            if a_qf != 0 or b_qf !=0 : continue


            year_t  = int(match.group(1))
            month_t = int(match.group(2))
            day_t   = int(match.group(3))
            hour_t  = int(match.group(4))
            min_t   = int(match.group(5))
            
            time_utc = datetime.datetime(year_t,month_t,day_t,hour_t,min_t,0)
            time_tai = time.Time(time_utc,format='datetime',scale='utc').tai.datetime
            flux_short = float(words[3])
            flux_long = float(words[6])
            goes = GOES(t_tai=time_tai, xray_flux_long=flux_long, xray_flux_short=flux_short)
            session.merge(goes)
            ctr+=1
    print 'commiting {} rows.'.format(ctr)
    session.commit()


for year in reversed(range(2011,2016)):
    for month in reversed(range(1,13)):
        if year==2015 and month > 8: continue
        (_, day_end) = calendar.monthrange(year,month)
        url = 'http://satdat.ngdc.noaa.gov/sem/goes/data/new_avg/{year}/{month:02d}/goes15/csv/g15_xrs_1m_{year}{month:02d}{day_begin:02d}_{year}{month:02d}{day_end:02d}.csv'.format(year=year,month=month,day_begin=1,day_end=day_end)

        print 'reading ' + url
        fp = urllib2.urlopen(url)
        con = fp.read()
        print 'parsing...'
        write_db(con)

