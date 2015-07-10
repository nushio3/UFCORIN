#!/usr/bin/env python

import argparse
import math
import scipy.ndimage.interpolation as intp
import numpy as np
import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import re, subprocess
import random
from astropy.io import fits

def system(cmd):
    subprocess.call(cmd, shell=True)

def gnuplot(cmd):
    with open('tmp.gnu','w') as fp:
        fp.write(cmd)
    subprocess.call('gnuplot tmp.gnu',shell=True)


#system('aws s3 sync s3://sdo/hmi/mag720/2015/07/ 07/')



def process(fn):
    n=1024
    n_original=4096
    n2=n_original/n
    pngfn = 'frames-{}/'.format(n) + fn.replace('/','-').replace('.fits','.png')
    binfn = 'scaled-{}/'.format(n) + fn.replace('/','-').replace('.fits','')
    print '{} -> {}'.format(fn, pngfn)

    system('mkdir -p frames-{}'.format(n))
    system('mkdir -p scaled-{}'.format(n))

    try:
        hdulist=fits.open(fn)
    except:
        return
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

    np.save(binfn, np.float32(img2))
#    print np.shape(img2)

    for y in range(n):
        for x in range(n):
            str+='{} {} {}\n'.format(x,y,img2[y][n-1-x])
        str+='\n'
    
    with open('test.txt','w') as fp:
        fp.write(str)
    
    gnuplot("""
set term png 20 size 1024,768
set out '{png}'
set pm3d
set pm3d map
set size ratio -1
set xrange [0:{n}]
set yrange [{n}:0]
set cbrange [-20:20]
red(x)=1-atan(3*(x-0.5))/pi*2
green(x)=1-10*(x-0.5)**2
blue(x)=1+atan(3*(x-0.5))/pi*2
set palette functions red(gray),green(gray),blue(gray)
splot 'test.txt' t '{title}'
    """.format(n=n,png=pngfn,title=fn.replace('.fits','')))


p=subprocess.Popen('find 07/ | grep .fits | sort',shell=True, stdout=subprocess.PIPE)
stdout, _ = p.communicate()

for fn in stdout.split('\n'):
    process(fn)
