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
        subprocess.call('time ./realtime-forecast.py --work-dir=. -r do',shell=True)
        subprocess.call('time ./review-forecast.py',shell=True)
        subprocess.call('cp review-forecast.png  ~/public_html',shell=True)

    
        subprocess.call('scp review-forecast.png ufcorin@kyoto-76483it5.cloudapp.net:/var/www/html/wordpress/wp-content/uploads/2016/04/review-forecast-1.png', shell=True)

    time.sleep(10)
