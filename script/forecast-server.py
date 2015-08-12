#!/usr/bin/env python

import datetime
import subprocess

epoch = datetime.datetime(2015,1,1)
now = datetime.datetime.now()

last_diff=None
while True:
    diff = int((now-epoch).total_seconds())//(60*12)
    if last_diff < diff:
        last_diff=diff
        subprocess.call('time ./realtime_forecast.py -r do',shell=True)
