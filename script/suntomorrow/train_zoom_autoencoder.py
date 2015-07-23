#!/usr/bin/env python
"""Chainer example: autoencoder of a solar image.
"""

# c.f.
# http://nonbiri-tereka.hatenablog.com/entry/2015/06/21/220506
# http://qiita.com/kenmatsu4/items/99d4a54d5a57405ecaf8

import argparse
from astropy.io import fits
import numpy as np
import scipy.ndimage.interpolation as intp
import operator
import os
import re
import six
import subprocess
import random
import pickle

import chainer
from chainer import computational_graph as c
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainer import optimizers

import matplotlib as mpl
mpl.use('Agg')
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


def plot_img(img4,fn,title_str):
    print np.shape(img4)

    if gpu_flag :
        img4 = cuda.to_cpu(img4)

    img=img4[0][0]

    fig, ax = plt.subplots()
	
	
    circle1=plt.Circle((512,512),450,edgecolor='black',fill=False)
	
    cmap = plt.get_cmap('bwr')
    cax = ax.imshow(img,cmap=cmap,extent=(0,1024,0,1024),vmin=-100.0,vmax=100.0)
    cbar=fig.colorbar(cax)
    fig.gca().add_artist(circle1)
    ax.set_title(title_str)
    fig.savefig('/home/ubuntu/public_html/{}.png'.format(fn))
    plt.close('all')

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

log_train_fn = 'log-training.txt'
log_test_fn = 'log-test.txt'

def system(cmd):
    subprocess.call(cmd, shell=True)

system('rm '+ log_train_fn)
system('rm '+ log_test_fn)

def zoom_x2(batch):
    shape = batch.data.shape
    channel_shape = shape[0:-2]
    height, width = shape[-2:]
 
    volume = reduce(operator.mul,shape,1)
 
    b1 = F.reshape(batch,(volume,1))
    b2 = F.concat([b1,b1],1)
 
    b3 = F.reshape(b2,(volume/width,2*width))
    b4 = F.concat([b3,b3],1)
 
    return F.reshape(b4, channel_shape + (2*height ,) + (2*width ,))


global gpu_flag
gpu_flag=(args.gpu >= 0)

global sun_data
sun_data = []

global dlDepth
dlDepth = 8

modelDict = dict()
for d in range(dlDepth):
    modelDict['convA{}'.format(d)] = F.Convolution2D( 2**d, 2**(d+1),3,stride=1,pad=1)
for d in range(dlDepth):
    modelDict['convV{}'.format(d)] = F.Convolution2D( 2**(d+1), 2**d,3,stride=1,pad=1)

model=chainer.FunctionSet(**modelDict)

if gpu_flag:
    cuda.init(0)
    model.to_gpu()

def layer_norm(x_data,level=1):
    global dlDepth
    x = Variable(x_data)

    hm = x
    for d in range(level):
        hc = (getattr(model,'convA{}'.format(d))(hm))
        if d < level - 1:
            hm = F.average_pooling_2d(hc,2)
    return F.mean_squared_error(hc,0*hc)

def forward(x_data,train=True,level=1):
    global dlDepth
    deploy = True
    x = Variable(x_data, volatile = not train)
    y = Variable(x_data, volatile = not train)

    hm = F.dropout(x, ratio = 0.1, train=deploy)
    for d in range(level):
        hc = (getattr(model,'convA{}'.format(d))(hm))
        if d < level - 1:
            hm = F.average_pooling_2d(hc,2)
        
    for d in reversed(range(level)):    
        if d < level - 1:        
            hc = zoom_x2(hm)
        hm =(getattr(model,'convV{}'.format(d))(hc))

    y_pred = hm

    ret = F.mean_squared_error(y,y_pred)
    if(not train):
        plot_img(y_pred.data, level, 'Lv {} autoencoder, msqe={}'.format(level, ret.data))
    if(not train and level==1):
        plot_img(y.data, 0, 'original magnetic field image')
    return ret


def reference(x_data,y_data):
    x = Variable(x_data)
    y = Variable(y_data)
    print "rmsqerr_adj: {}".format(F.mean_squared_error(x,y).data)


def load_fits(fn):
    n=1024
    n_original=4096
    n2=n_original/n
    print 'now loading {}'.format(fn)

    try:
        hdulist=fits.open(fn)
    except:
        return []
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

    return [np.float32(img2)]

def fetch_data():
    global sun_data

    system('rm work/*')
    while not os.path.exists('work/0000.npz'):
        y=random.randrange(2015,2016)
        m=random.randrange(1,13)
        d=random.randrange(1,32)
        cmd='aws s3 sync --quiet s3://sdo/hmi/mag720x1024/{:04}/{:02}/{:02}/ work/'.format(y,m,d)
        system(cmd)


    p=subprocess.Popen('find work/',shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    sun_data = []

    for fn in stdout.split('\n'):
        if not re.search('\.npz$',fn) : continue
        try:
            sun_data.append(np.load(fn)['img'])
        except:
            continue

# if len(sun_data)==0:
#     # where no data is available, add a dummy data for debugging
#     for i in range(10):
#         x=32*[0.333*i*i]
#         xy=32*[x]
#         sun_data.append(xy)



optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())
optimizer_norm = optimizers.Adam()
optimizer_norm.setup(model.collect_parameters())




epoch=0
while True:
    fetch_data()
    reference(np.array(sun_data[0]), np.array(sun_data[1]))

    for t in range(20): # use the same dataset 
      epoch+=1

      for level in range(1,dlDepth+1):
        batch= []
    
    
        for i in range(4):
            start = random.randrange(len(sun_data))
            batch.append([sun_data[start]])
    
        batch=np.array(batch)
        if gpu_flag :
            batch = cuda.to_gpu(batch)

        batch_norm = F.mean_squared_error(Variable(batch),0*Variable(batch))
        if gpu_flag :
            batch_norm = cuda.to_gpu(batch_norm)
        batch_norm = float(str(batch_norm.data))
        

    
        optimizer_norm.zero_grads()
        this_layer_norm = layer_norm(batch, level=level)
        loss = (this_layer_norm - batch_norm)**2
        loss.backward()
        optimizer_norm.update()
        if epoch % 10 == 1:
            print 'normalization --- {} : {}'.format(this_layer_norm.data, batch_norm)

        optimizer.zero_grads()
        loss = forward(batch, train=True,level=level)
        loss.backward()
        optimizer.update()


        print '  '*(level-1),epoch,loss.data
    
        with(open(log_train_fn,'a')) as fp:
            fp.write('{} {} {}\n'.format(level,epoch,loss.data))
    
        if epoch == 1:
            with open("graph{}.dot".format(level), "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph{}.wo_split.dot".format(level), "w") as o:
                g = c.build_computational_graph((loss, ),
                                                    remove_split=True)
                o.write(g.dump())
            print('graph generated')
    
        if epoch % 10 == 1:
            loss = forward(batch,train=False,level=level)
            print "T",'  '*(level-1),epoch,loss.data
            with(open(log_test_fn,'a')) as fp:
                fp.write('{} {} {}\n'.format(level, epoch,loss.data))
