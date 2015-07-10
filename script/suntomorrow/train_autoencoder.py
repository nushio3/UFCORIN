#!/usr/bin/env python
"""Chainer example: autoencoder of a solar image.
"""

# c.f. 
# http://nonbiri-tereka.hatenablog.com/entry/2015/06/21/220506
# http://qiita.com/kenmatsu4/items/99d4a54d5a57405ecaf8

import argparse

import numpy as np
import re
import six
import subprocess
import random

import chainer
from chainer import computational_graph as c
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from chainer import optimizers



gpu_flag=True

# load the numpy 2D arrays located under the folder.
p=subprocess.Popen('find scaled-256/',shell=True, stdout=subprocess.PIPE)
stdout, _ = p.communicate()

sun_data = []

for fn in stdout.split('\n'):
    if not re.search('\.npy$',fn) : continue
    # put each image into [] because it is the only channel.
    sun_data.append([np.load(fn)])

conv1 = F.Convolution2D(1,1,3,stride=1,pad=1)
conv2 = F.Convolution2D(1,1,3,stride=1,pad=1)
conv3 = F.Convolution2D(1,1,3,stride=1,pad=1)

model=chainer.FunctionSet(l1 = conv1, l2 = conv2, l3 = conv3)

if gpu_flag:
    cuda.init(0)
    model.to_gpu()



def forward(x_data,train=True):
    x = Variable(x_data, volatile = not train)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.tanh(model.l2(h1)), train=train)
    h3 = model.l3(h2)
    return F.mean_squared_error(x,h3)


optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters()) 

for epoch in range(1000000):
    batch = []
    for i in range(50):
        batch.append(random.choice(sun_data))
    batch=np.array(batch)
    if gpu_flag : 
        batch = cuda.to_gpu(batch)

    optimizer.zero_grads()
    loss = forward(batch, train=True)
    loss.backward()
    optimizer.update()
    
    print epoch,loss.data

    with(open('train-log.txt','a')) as fp:
        fp.write('{} {}\n'.format(epoch,loss.data))






