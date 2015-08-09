#!/usr/bin/env python

import datetime, glob, os, re, shutil, subprocess, sys
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


with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()


def system(cmd):
    subprocess.call(cmd, shell=True)

path= '/home/ubuntu/hub/UFCORIN/script/jsoc/'

workdir="nrt"
if not os.path.exists(workdir): os.mkdir(workdir)
os.chdir(workdir)





# Download the latest NRT FITS image

series_name = "hmi.M_720s_nrt"
query = series_name + "[2015.08.01-2015.08.09@720s]"
command = path+"exportfile.csh '"+query+ "' " + sys.argv[1]
print command
system("rm *.fits")
system(command)

# create the DB class for our particular wavelet
DB = wavelet.db_class(series_name, 'haar')
engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))
DB.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()



def fits2npz(newfn, npzfn):
    hdulist=fits.open(newfn)
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
    

for fn in glob.glob('*.fits'):
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
    t=t_tai.utc.datetime
    print "got data at {}".format(t)

    newfn='{:02}{:02}.fits'.format(hh,minu)
    shutil.copy(fn,newfn)
    s3 = "aws s3 cp "+newfn+" s3://sdo/hmi/mag720/{:04}/{:02}/{:02}/{}".format(yyyy,mm,dd,newfn)
    print s3
    system(s3)

    reso_original=4096
    reso_new=1024
    zoom_ratio = float(reso_new)/reso_original

    npzfn = newfn.replace('.fits','.npz')
    img = fits2npz(newfn, npzfn)

    cmd='aws s3 cp '+npzfn+ (' s3://sdo/hmi/mag720x{}/{:04}/{:02}/{:02}/'.format(reso_new,yyyy,mm,dd))+npzfn
    print cmd
    system(cmd)

    r = DB()
    r.fill_columns(t, img)

    session.merge(r)
    session.commit()
