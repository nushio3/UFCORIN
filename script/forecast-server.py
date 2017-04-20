#!/usr/bin/env python

import datetime,time,sys
import subprocess
from functools import wraps

def on_timeout(limit, handler, hint=None):
    '''                                                 
    call handler with a hint on timeout(seconds)
    http://qiita.com/siroken3/items/4bb937fcfd4c2489d10a
    '''
    def notify_handler(signum, frame):
        handler("'%s' is not finished in %d second(s)." % (hint, limit))

    def __decorator(function):
        def __wrapper(*args, **kwargs):
            import signal
            signal.signal(signal.SIGALRM, notify_handler)
            signal.alarm(limit)
            result = function(*args, **kwargs)
            signal.alarm(0)
            return result
        return wraps(function)(__wrapper)
    return __decorator
def abort_handler(msg):
    global child_proc
    sys.stderr.write(msg)
    child_proc.kill()
    sys.exit(1)


@on_timeout(limit=700, handler = abort_handler, hint='realtime forecast')
def realtime_forecast():
    global child_proc
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    child_proc = subprocess.Popen('exec ./realtime-forecast.py --work-dir=. -r do',shell=True)
    child_proc.wait()

epoch = datetime.datetime(2015,1,1)

last_diff=None
while True:
    now = datetime.datetime.now()
    diff = int((now-epoch).total_seconds())//(60*12)
    if last_diff < diff:
        last_diff=diff
        try:
            realtime_forecast()
        except Exception as e:
            print e.message
            pass
        subprocess.call('time ./review-forecast.py',shell=True)
        subprocess.call('cp review-forecast.png  ~/public_html',shell=True)

    
        subprocess.call('scp review-forecast.png ufcorin@kyoto-76483it5.cloudapp.net:/var/www/html/wordpress/wp-content/uploads/2016/04/review-forecast-1.png', shell=True)

    time.sleep(10)
