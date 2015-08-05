#!/usr/bin/env python

import datetime
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

with open('.mysqlpass','r') as fp:
    password = fp.read().strip()

Base = declarative_base()

class GOES(Base):
    __tablename__ = 'goes_xray_flux'

    t = sql.Column(sql.DateTime, primary_key=True)
    xray_flux_long = sql.Column(sql.Float)
    xray_flux_short = sql.Column(sql.Float)


goes = GOES(t=datetime.datetime(2025,8,5,8,50,0), xray_flux_long=2.335e-6, xray_flux_short=1.555e-7)



engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))
# GOES.metadata.create_all(engine)


Session = sessionmaker(bind=engine)
session = Session()

session.merge(goes)
session.commit()
