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

system('mkdir -p frames')

def process(fn):
    pngfn = 'frames/' + fn.replace('/','-').replace('.fits','.png')
    print '{} -> {}'.format(fn, pngfn)

    try:
        hdulist=fits.open(fn)
    except:
        return
    img=hdulist[1].data
    str = ''
    n=1024
    n2=4096/n

    img2=intp.zoom(img,zoom=1.0/n2)
#    print np.shape(img2)

    for y in range(n):
        for x in range(n):
            str+='{} {} {}\n'.format(x,y,img2[y][n-1-x])
        str+='\n'
    
    with open('test.txt','w') as fp:
        fp.write(str)
    
    gnuplot("""
set term png 20 size 2048,2048
set out '{png}'
set pm3d
set pm3d map
set size ratio -1
set xrange [0:{n}]
set yrange [{n}:0]
set cbrange [-3:3]
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
