#!/usr/bin/env python

import argparse, datetime, glob, os, pickle, re, shutil, subprocess, signal, sys, traceback
from astropy.io import fits
import astropy.time as time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation as intp
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import wavelet

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

class WatchState:
    last_success_time = None
    last_cached_time = None

parser = argparse.ArgumentParser(description='HMI Downloader and converter')
parser.add_argument('--convert', '-c', action='store_true',
                    help='convert instead of downloading the latest data')
parser.add_argument('--mail-address', '-m', type=str,
                    help='mail address registered at JSOC')
args = parser.parse_args()


with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()

original_working_directory = os.getcwd()
watch_state=WatchState()
try:
    with open('download-hmi-nrt.state','r') as fp:
        watch_state = pickle.load(fp)
except:
    pass


@on_timeout(limit=3600, handler = abort_handler, hint='system call')
def system(cmd):
    global child_proc
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    child_proc = subprocess.Popen("exec " + cmd, shell=True)
    child_proc.wait()

path= '/home/ubuntu/hub/UFCORIN/script/jsoc/'

if not args.convert:
    workdir="nrt"
    if not os.path.exists(workdir): os.mkdir(workdir)
    system("rm "+workdir+"/*")
    os.chdir(workdir)
else:
    workdir="convert"
    if not os.path.exists(workdir): os.mkdir(workdir)
    system("rm "+workdir+"/*")
    os.chdir(workdir)




# Download the latest NRT FITS image

print "last success " , watch_state.last_success_time
series_name = "hmi.M_720s_nrt"
query = series_name + "[2015.12.01_00:00:00-2015.12.02_03:00:00]"
if watch_state.last_success_time:
    watch_state.last_cached_time += time.TimeDelta(3600, format='sec')
    t_begin = watch_state.last_success_time.datetime + datetime.timedelta(minutes=1)
    t_end   = watch_state.last_cached_time.datetime  + datetime.timedelta(hours=24)
    query = series_name + '[{}-{}]'.format(
        t_begin.strftime('%Y.%m.%d_%H:%M:%S'),
        t_end.strftime('%Y.%m.%d_%H:%M:%S') )

command = path+"exportfile.csh '"+query+ "' " + args.mail_address
if not args.convert:
    print command
    system(command)

# create the DB class for our particular wavelet
DB = wavelet.db_class(series_name, 'haar')
engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))
DB.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()



def fits2npz(newfn, npzfn):
    hdulist=fits.open(newfn)
    hdulist.verify('fix')
    img=hdulist[1].data
    img = np.where( np.isnan(img), 0.0, img)
    img2=intp.zoom(img,zoom=zoom_ratio)

    for y in range(reso_new):
        for x in range(reso_new):
            x0=reso_new/2.0-0.5
            y0=reso_new/2.0-0.5
            r2 = (x-x0)**2 + (y-y0)**2
            r0 = 1800.0*zoom_ratio
            if r2 >= r0**2 : img2[y][x]=0.0
    img32 = np.float32(img2)
    np.savez_compressed(npzfn, img=img32)

    return img32


def plot_img(img,fn,title_str):
    w,h= np.shape(img)
    dpi=200
    plt.figure(figsize=(8,6),dpi=dpi)
    fig, ax = plt.subplots()

    cmap = plt.get_cmap('bwr')
    cax = ax.imshow(img,cmap=cmap,vmin=-100.0,vmax=100.0)  # extent=(0,w,0,h),
    cbar=fig.colorbar(cax)
    ax.set_title(title_str)
    fig.savefig(fn,dpi=dpi)
    plt.close('all')


def register_wavelet(img, imgfn):
    plot_img(img,imgfn,"real")
    wavelet_img = wavelet.wavedec2_img(img,'haar','NS')
    plot_img(wavelet_img,"NS_" + imgfn,"NS-wavelet")
    wavelet_img = wavelet.wavedec2_img(img,'haar','S')
    plot_img(wavelet_img,"S_" + imgfn,"S-wavelet")

if args.convert:
    for yyyy in reversed(range(2011,2016)):
        for mm in reversed(range(1,13)):
            if yyyy==2015 and mm >=8 : continue
            for dd in reversed(range(1,32)):
                system("rm *")
                s3 = "aws s3 sync s3://sdo/hmi/mag720x1024/{:04}/{:02}/{:02}/ .".format(yyyy,mm,dd)
                print s3
                system(s3)
                for fn in glob.glob('*.npz'):
                    ma=re.search('(\d+)',fn)
                    if not ma: continue
                    hh=int(ma.group(1)[0:2])
                    minu=int(ma.group(1)[2:4])
                    t_tai = time.Time('{:04}-{:02}-{:02} {:02}:{:02}'.format(yyyy,mm,dd,hh,minu),scale='tai', format='iso')
                    print "got data at {}".format(t_tai)
                    r = DB()
                    try:
                        img=np.load(fn)['img']
                        r.fill_columns(t_tai.datetime, img)
                        session.merge(r)
                    except:
                        pass
                session.commit()
    exit(0)

for fn in sorted(glob.glob('*.fits')):
    if not re.match('hmi',fn): continue

    ma = re.search('%5B(\d+)\.(\d+)\.(\d+)_(\d+)%3A(\d+)',fn)
    if not ma: continue
    yyyy=int(ma.group(1))
    mm  =int(ma.group(2))
    dd  =int(ma.group(3))
    hh  =int(ma.group(4))
    minu=int(ma.group(5))

    # convert TAI to UTC
    t_tai = time.Time('{:04}-{:02}-{:02} {:02}:{:02}'.format(yyyy,mm,dd,hh,minu),scale='tai', format='iso')

    print "got data at {} ".format(t_tai)

    newfn='{:02}{:02}.fits'.format(hh,minu)
    shutil.copy(fn,newfn)
    s3 = "aws s3 cp "+newfn+" s3://sdo/hmi/mag720/{:04}/{:02}/{:02}/{}".format(yyyy,mm,dd,newfn)
    print s3
    system(s3)

    reso_original=4096
    reso_new=1024
    zoom_ratio = float(reso_new)/reso_original

    try:
        npzfn = newfn.replace('.fits','.npz')
        img = fits2npz(newfn, npzfn)

        cmd='aws s3 cp '+npzfn+ (' s3://sdo/hmi/mag720x{}/{:04}/{:02}/{:02}/'.format(reso_new,yyyy,mm,dd))+npzfn
        print cmd
        system(cmd)

        r = DB()
        r.fill_columns(t_tai.datetime, img)

        session.merge(r)
        session.commit()

        if not watch_state.last_success_time or t_tai > watch_state.last_success_time:
            watch_state.last_success_time = t_tai
            watch_state.last_cached_time = t_tai
    except Exception as e:
        print traceback.format_exc()
        print e

os.chdir(original_working_directory)
with open('download-hmi-nrt.state','w') as fp:
    pickle.dump(watch_state,fp)
