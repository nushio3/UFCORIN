#!/usr/bin/env python

import psycopg2
import datetime
import pytz
import random

conn = psycopg2.connect('dbname=sunfeature user=ufcoroot host=ufcorin-postgresql.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com')
cur = conn.cursor()

#cur.execute('SELECT * FROM goes;')
#print cur.fetchone()

for s in range(60):
    t = datetime.datetime(2012, 12, 03, 03, 21, s,0,pytz.utc)
    cur.execute('INSERT INTO goes (t,x_ray_flux_long) VALUES (%s, %s) ON CONFLICT DO UPDATE;', (t,random.random()))

conn.commit()

cur.close()
conn.close()
