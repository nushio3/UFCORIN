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

from datetime import datetime



parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()


def system(cmd):
    subprocess.call(cmd, shell=True)


global work_dir
work_dir='/home/ubuntu/public_html/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

system('mkdir -p ' + work_dir)

log_train_fn = work_dir + '/log-training.txt'
log_test_fn = work_dir + '/log-test.txt'

system('cp {} {} '.format(__file__, work_dir))

def plot_img(img4,fn,title_str):
    global  global_normalization, work_dir
    print np.shape(img4)

    if gpu_flag :
        img4 = cuda.to_cpu(img4)

    img=(1.0/ global_normalization)*img4[0][0]

    fig, ax = plt.subplots()
	
	
    circle1=plt.Circle((512,512),450,edgecolor='black',fill=False)
	
    cmap = plt.get_cmap('bwr')
    cax = ax.imshow(img,cmap=cmap,extent=(0,1024,0,1024),vmin=-100.0,vmax=100.0)
    cbar=fig.colorbar(cax)
    fig.gca().add_artist(circle1)
    ax.set_title(title_str)
    fig.savefig('{}/{}.png'.format(work_dir,fn))
    plt.close('all')

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

global sun_data,  global_normalization
sun_data = []

global dlDepth
dlDepth = 6
global_normalization = 1e-3

modelDict = dict()
for d in range(dlDepth):
    modelDict['convA{}'.format(d)] = F.Convolution2D( 2**d, 2**(d+1),3,stride=1,pad=1)
    modelDict['convB{}'.format(d)] = F.Convolution2D( 2**(d+1), 2**(d+1),3,stride=1,pad=1)
    modelDict['convV{}'.format(d)] = F.Convolution2D( 2**(d+1), 2**d,3,stride=1,pad=1)

model=chainer.FunctionSet(**modelDict)

if gpu_flag:
    cuda.init(0)
    model.to_gpu()

def sigmoid2(x):
    return F.sigmoid(x)*2.0-1.0

def forward_dumb(x_data,train=True,level=1):
    x = Variable(x_data)
    y = Variable(x_data)
    for d in range(level):    
        x = F.average_pooling_2d(x,2)
    for d in range(level):    
        x = zoom_x2(x)

    ret = (global_normalization**(-2))*F.mean_squared_error(sigmoid2(y),sigmoid2(x))
    if(not train):
        plot_img(x.data, 'd{}'.format(level), 
                 'Lv {} dumb encoder, msqe={}'.format(level, ret.data))
    return ret

        

def forward(x_data,train=True,level=1):
    global dlDepth
    deploy = train
    x = Variable(x_data, volatile = not train)
    y = Variable(x_data, volatile = not train)

    h = F.dropout(x, ratio = 0.1, train=deploy)
    for d in range(level):
        h = sigmoid2(getattr(model,'convA{}'.format(d))(h))
        if d < level - 1:
            h = F.dropout(h, ratio = 0.1, train=deploy)
        h = F.max_pooling_2d(h,2)

    h = sigmoid2(getattr(model,'convB{}'.format(level-1))(h))
        
    for d in reversed(range(level)):    
        h = zoom_x2(h)
        h = sigmoid2(getattr(model,'convV{}'.format(d))(h))

    y_pred = h

    ret = (global_normalization**(-2))*F.mean_squared_error(sigmoid2(y),y_pred)
    if(not train):
        plot_img(y_pred.data, level, 'Lv {} autoencoder, msqe={}'.format(level, ret.data))
    if(not train and level==1):
        plot_img(y.data, 0, 'original magnetic field image')
    return ret


def reference(x_data,y_data):
    global global_normalization
    x = Variable(x_data)
    y = Variable(y_data)
    print "rmsqerr_adj: {}".format((global_normalization**(-2))*F.mean_squared_error(x,y).data)



def fetch_data():
    global sun_data, global_normalization

    system('rm work/*')
    while not os.path.exists('work/0000.npz'):
        y=random.randrange(2011,2016)
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
            sun_data.append( global_normalization*np.load(fn)['img'])
        except:
            continue

# if len(sun_data)==0:
#     # where no data is available, add a dummy data for debugging
#     for i in range(10):
#         x=32*[0.333*i*i]
#         xy=32*[x]
#         sun_data.append(xy)


optimizer = dict()
for level in range(1,dlDepth+1):
    optimizer[level] = optimizers.Adam() #(alpha=3e-4)
    d=level-1
    model_of_level=dict()
    k='convA{}'.format(d)
    model_of_level[k]=modelDict[k]
    k='convB{}'.format(d)
    model_of_level[k]=modelDict[k]
    k='convV{}'.format(d)
    model_of_level[k]=modelDict[k]
    optimizer[level].setup(chainer.FunctionSet(**model_of_level).collect_parameters())


epoch=0
while True:
    fetch_data()
    try:
        reference(np.array(sun_data[0]), np.array(sun_data[1]))
    except:
        continue

    for t in range(20): # use the same dataset 
      epoch+=1

      batch= []
    
    
      for i in range(3):
          start = random.randrange(len(sun_data))
          batch.append([sun_data[start]])
    
      batch=np.array(batch)
      if gpu_flag :
            batch = cuda.to_gpu(batch)

      current_depth = min(dlDepth+1,max(2,2+epoch/2000))
      current_depth = dlDepth+1

      for level in range(1,current_depth):
        if level < current_depth-1:
            optimizer[level].alpha=1e-4
        optimizer[level].zero_grads()
        loss = forward(batch, train=True,level=level)
        loss.backward()
        optimizer[level].update()


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
            loss = forward_dumb(batch, train=False,level=level)
            loss_dumb = loss.data
            loss = forward(batch,train=False,level=level)
            loss_auto = loss.data
            print "T",'  '*(level-1),epoch,loss_auto, loss_auto/loss_dumb
            with(open(log_test_fn,'a')) as fp:
                fp.write('{} {} {} {}\n'.format(level, epoch,loss_auto, loss_dumb))
