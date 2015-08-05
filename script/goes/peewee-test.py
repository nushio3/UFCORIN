#!/usr/bin/env python

import peewee
import datetime
from peewee import *

pstr = ''

with open('.mysqlpass','r') as fp:
    pstr = fp.read().strip()



db = MySQLDatabase('sun_feature',host='sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com', user='ufcoroot',passwd=pstr,port=3306)

class GoesLightCurve(peewee.Model):
    t = peewee.DateTimeField(unique=True)
    x_ray_flux_long = peewee.FloatField()

    class Meta:
        database = db

# GoesLightCurve.create_table()
goes = GoesLightCurve(t=datetime.datetime(2025,8,5,8,50,0), x_ray_flux_long=2.333e-6)
goes.save(force_insert=True)
for goes in GoesLightCurve.select():
    print goes.x_ray_flux_long
