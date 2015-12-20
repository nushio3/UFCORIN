#!/usr/bin/env python

import pickle,argparse,os,subprocess,sys
import numpy as np
from PIL import Image
from StringIO import StringIO
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage



import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--fresh-start', '-f', action='store_true',
                    help='Start simulation anew')
args = parser.parse_args()


nz = 100          # # of dim for final layer
batchsize=25
patch_pixelsize=128
n_epoch=10000
n_train=200000
save_interval =20000

n_timeseries = 4

out_image_dir = './out_images_%s'%(args.gpu)
out_model_dir = './out_models_%s'%(args.gpu)


subprocess.call("mkdir -p %s "%(out_image_dir),shell=True)
subprocess.call("mkdir -p %s "%(out_model_dir),shell=True)


################################################################
## Additional Combinators
################################################################

class ELU(function.Function):

    """Exponential Linear Unit."""
    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,


def elu(x, alpha=1.0):
    """Exponential Linear Unit function."""
    # https://github.com/muupan/chainer-elu
    return ELU(alpha=alpha)(x)


################################################################
## Neural Networks
################################################################

class Evolver(chainer.Chain):
    def __init__(self):
        super(Evolver, self).__init__(
            c0 = L.Convolution2D(n_timeseries-1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            c4 = L.Convolution2D(512, 1024, 8, stride=1, pad=0, wscale=0.02*math.sqrt(4*4*512)),
            dc4 = L.Deconvolution2D(1024,512, 8, stride=1, pad=0, wscale=0.02*math.sqrt(4*4*1024)),
            dc3 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc1 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc0 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
            bn4 = L.BatchNormalization(1024)
        )
        
    def __call__(self, x, test=False):
        h48 = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h24 = elu(self.bn1(self.c1(h48), test=test))
        h12 = elu(self.bn2(self.c2(h24), test=test))
        h6 = elu(self.bn3(self.c3(h12), test=test))
        h1 = elu(self.bn4(self.c4(h6), test=test))
        
        # idea: not simple addition, but concatenation?
        h = h6 + 1e-1*F.relu(self.bn3(self.dc4(h1), test=test))
        h = h12 + 1e-1*F.relu(self.bn2(self.dc3(h), test=test))
        h = h24 + 1e-1*F.relu(self.bn1(self.dc2(h), test=test))
        h = h48 + 1e-1*F.relu(self.bn0(self.dc1(h), test=test))
        ret=self.dc0(h)
        
        #return ret

        return F.split_axis(x,[1],1)[0]+ret
        #return Variable(xp.zeros((batchsize, 1, 128,128), dtype=np.float32))







class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(n_timeseries, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(8*8*512, 2, wscale=0.02*math.sqrt(8*8*512)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        h = elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l





################################################################
## Main program
################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--fresh-start', '-f', action='store_true',
                    help='Start simulation anew')
args = parser.parse_args()



def load_dataset():
    pw=patch_pixelsize
    ph=patch_pixelsize

    ret_in  = np.zeros((batchsize, n_timeseries-1, ph, pw), dtype=np.float32)
    ret_out = np.zeros((batchsize, 1, ph, pw), dtype=np.float32)

    img = n_timeseries*[None]
    img[0] = np.load('aia193/0012.npz')['img']
    img[1] = np.load('aia193/0024.npz')['img']
    img[2] = np.load('aia193/0036.npz')['img']
    img[3] = np.load('aia193/0048.npz')['img']

    for j in range(batchsize):
        oh, ow = img[0].shape
        left  = np.random.randint(ow-pw)
        top   = np.random.randint(oh-ph)
        for t in range(n_timeseries):
            if t==n_timeseries -1:
                ret_out[j,0,:,:]=img[t][top:top+ph, left:left+pw]
            else:
                ret_in[j,t,:,:]=img[t][top:top+ph, left:left+pw]
    print "input flags"
    print ret_in.flags.c_contiguous
    print ret_out.flags.c_contiguous
    return (ret_in, ret_out)



def train_dcgan_labeled(evol, dis, epoch0=0):
    o_evol = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_evol.setup(evol)
    o_dis.setup(dis)
    if not args.fresh_start:
        serializers.load_hdf5("%s/dcgan_model_dis.h5"%(out_model_dir),dis)
        serializers.load_hdf5("%s/dcgan_model_evol.h5"%(out_model_dir),evol)
        serializers.load_hdf5("%s/dcgan_state_dis.h5"%(out_model_dir),o_dis)
        serializers.load_hdf5("%s/dcgan_state_evol.h5"%(out_model_dir),o_evol)


    o_evol.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    for epoch in xrange(epoch0,n_epoch):
        print "epoch:", epoch
        
        for i in xrange(0, n_train, batchsize):
            print "train:",i
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            data_in, data_out = load_dataset()
            movie_in = Variable(cuda.to_gpu(data_in))
            movie_out = Variable(cuda.to_gpu(data_out))
            movie_train = F.concat([movie_in, movie_out])
            #print "load image done"
            
            movie_out_predict = evol(movie_in)
            movie_predict = F.concat([movie_in, movie_out_predict])
            yl = dis(movie_predict)


            print yl.data.shape

            L_evol = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis  = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))

            # train discriminator
            yl_train = dis(movie_train)
            L_dis += F.softmax_cross_entropy(yl_train, Variable(xp.zeros(batchsize, dtype=np.int32)))
            

            print "mp",movie_predict.data.flags.c_contiguous
            print "yl",yl.data.flags.c_contiguous
            print "Le",L_evol.data.flags.c_contiguous
            print "Le",L_evol.data.shape
            
            
            o_evol.zero_grads()
            L_evol.backward()
            o_evol.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()



xp = cuda.cupy
cuda.get_device(int(args.gpu)).use()

evol = Evolver()
dis = Discriminator()
evol.to_gpu()
dis.to_gpu()

train_dcgan_labeled(evol,dis)
