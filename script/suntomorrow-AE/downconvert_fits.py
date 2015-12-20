#!/usr/bin/env python

import argparse
import math,os,sys
import scipy.ndimage.interpolation as intp
import numpy as np

import re, subprocess
import random
from astropy.io import fits

def system(cmd):
    subprocess.call(cmd, shell=True)

def load_fits(fn):
    print fn ,
    sys.stdout.flush()
    n=1024
    n_original=4096
    n2=n_original/n

    binfn = fn.replace('.fits','')

    try:
        hdulist=fits.open(fn)
        img=hdulist[1].data
        str = ''
    
        img = np.where( np.isnan(img), 0.0, img)
        
        img2=intp.zoom(img,zoom=1.0/n2)
    
        for y in range(n):
            for x in range(n):
                x0=n/2.0-0.5
                y0=n/2.0-0.5
                r2 = (x-x0)**2 + (y-y0)**2
                r0 = 1800.0*n/n_original
                if r2 >= r0**2 : img2[y][x]=0.0
    
        np.savez_compressed(binfn, img=np.float32(img2))
    except:
        return 


def fetch_data(y,m,d):
    global sun_data

    system('rm -fr work/*')
    cmd='aws s3 sync s3://sdo/hmi/mag720/{:04}/{:02}/{:02}/ work/'.format(y,m,d)
    print cmd
    system(cmd)

    if not os.path.exists('work/0000.fits'):
        return

    print 'converting: '
    p=subprocess.Popen('find work/ | sort',shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    for fn in stdout.split('\n'):
        if not re.search('\.fits$',fn) : continue
        load_fits(fn)
    system('rm work/*.fits')

    cmd='aws s3 sync  work/ s3://sdo/hmi/mag720x1024/{:04}/{:02}/{:02}/'.format(y,m,d)
    print cmd
    system(cmd)


# for y in reversed(range(2011,2016)):
#     for m in reversed(range(1,13)): 
#         if (y==2015 and m>=8): continue
#         for d in reversed(range(1,32)):
#             fetch_data(y,m,d)
# 

for d in reversed(range(1,32)):
    fetch_data(2015,7,d)
