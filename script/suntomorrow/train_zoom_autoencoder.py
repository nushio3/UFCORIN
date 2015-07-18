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



gpu_flag=(args.gpu >= 0)

global sun_data
sun_data = []




model=chainer.FunctionSet(
    convA1 = F.Convolution2D( 1, 2,3,stride=1,pad=1),
    convA2 = F.Convolution2D( 2, 4,3,stride=1,pad=1),
    convA3 = F.Convolution2D( 4, 8,3,stride=1,pad=1),
    convV3 = F.Convolution2D( 8, 4,3,stride=1,pad=1),
    convV2 = F.Convolution2D( 4, 2,3,stride=1,pad=1),
    convV1 = F.Convolution2D( 2, 1,3,stride=1,pad=1),
)

if gpu_flag:
    cuda.init(0)
    model.to_gpu()



def forward(x_data,train=True,level=1):
    deploy = True
    x = Variable(x_data, volatile = not train)
    y = Variable(x_data, volatile = not train)

    noisy_x = F.dropout(x, ratio = 0.1, train=deploy)
    hc1 = model.convA1(noisy_x)
    hm1 = F.average_pooling_2d(hc1,2)
    if level >= 2:
        hc2 = model.convA2(hm1)
        hm2 = F.average_pooling_2d(hc2,2)
        if level >= 3:
            hc3 = model.convA3(hm2)
            hm3 = F.average_pooling_2d(hc3,2)
        
            hz3 = zoom_x2(hm3)
            hv3 = model.convV3(hz3)
        else:
            hv3 = hm2
        hz2 = zoom_x2(hv3)
        hv2 = model.convV2(hz2)
    else:
        hv2=hm1
    hz1 = zoom_x2(hv2)
    hv1 = model.convV1(hz1)


    y_pred = hv1
    return F.mean_squared_error(y,y_pred)

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
    while not os.path.exists('work/0000.npy'):
        y=random.randrange(2011,2016)
        m=random.randrange(1,13)
        d=random.randrange(1,32)
        cmd='aws s3 sync --quiet s3://sdo/hmi/mag720x1024/{:04}/{:02}/{:02}/ work/'.format(y,m,d)
        system(cmd)


    p=subprocess.Popen('find work/',shell=True, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    sun_data = []

    for fn in stdout.split('\n'):
        if not re.search('\.npy$',fn) : continue
        sun_data.append(np.load(fn))

# if len(sun_data)==0:
#     # where no data is available, add a dummy data for debugging
#     for i in range(10):
#         x=32*[0.333*i*i]
#         xy=32*[x]
#         sun_data.append(xy)



optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())




epoch=0
while True:
    fetch_data()
    reference(np.array(sun_data[0]), np.array(sun_data[1]))

    epoch+=1
    for level in range(3):
        batch= []
    
    
        for i in range(10):
            start = random.randrange(len(sun_data))
            batch.append([sun_data[start]])
    
        batch=np.array(batch)
        if gpu_flag :
            batch = cuda.to_gpu(batch)
    
        optimizer.zero_grads()
        loss = forward(batch, train=True,level=level)
        loss.backward()
        optimizer.update()

        print '\t'*level,epoch,loss.data
    
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
            print "T",'\t'*level,epoch,loss.data
            with(open(log_test_fn,'a')) as fp:
                fp.write('{} {} {}\n'.format(level, epoch,loss.data))
