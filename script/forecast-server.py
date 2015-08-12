#!/usr/bin/env python

import datetime,time
import subprocess

epoch = datetime.datetime(2015,1,1)

last_diff=None
while True:
    now = datetime.datetime.now()
    diff = int((now-epoch).total_seconds())//(60*12)
    if last_diff < diff:
        last_diff=diff
        subprocess.call('time ./realtime_forecast.py -r do',shell=True)
    time.sleep(10)
