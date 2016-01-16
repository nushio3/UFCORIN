#!/usr/bin/env python
import pickle, argparse,urllib,re, subprocess
import scipy.ndimage.interpolation as intp


import numpy as np
from PIL import Image
import os
from StringIO import StringIO
import math
import pylab
from astropy.io import fits

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
# parser.add_argument('--gamma', default=1.0,type=float,
#                     help='weight of content similarity over style similarity')
# parser.add_argument('--creativity-weight', default=1.0,type=float,
#                     help='weight of creativity over emulation')
# parser.add_argument('--stride','-s', default=4,type=int,
#                     help='stride size of the final layer')
# parser.add_argument('--nz', default=100,type=int,
#                     help='the size of encoding space')
# parser.add_argument('--dropout', action='store_true',
#                     help='use dropout when training dis.')
# parser.add_argument('--enc-norm', default = 'dis',
#                     help='use (dis/L2) norm to train encoder.')

args = parser.parse_args()

xp = cuda.cupy
cuda.get_device(args.gpu).use()

def foldername(args):
    x = urllib.quote(str(args))
    x = re.sub('%..','_',x)
    x = re.sub('Namespace_','DCGAN_',x)
    return x

work_image_dir = '/mnt/work-{}'.format(args.gpu)
out_image_dir = '/mnt/public_html/out-images-{}'.format(foldername(args))
out_image_show_dir = '/mnt/public_html/out-images-{}'.format(args.gpu)
out_model_dir = './out-models-{}'.format(args.gpu)



img_w=1024
img_h=1024

patch_w=96
patch_h=96

nz = 100          # # of dim for Z
batchsize=100
n_epoch=10000
n_train=100000
image_save_interval = 20000

# read an image
def scale_brightness(x):
    lo = 50.0
    hi = 1250.0
    x2 = np.minimum(hi,np.maximum(lo,x))
    x3 = (np.log(x2)-np.log(lo)) / (np.log(hi) - np.log(lo))
    return x3

# c.f. SDO/AIA official pallette https://darts.jaxa.jp/pub/ssw/sdo/aia/idl/pubrel/aia_lct.pro

def data_to_image(i,var):
    img = var[i,0]
    img = np.maximum(0.0, np.minimum(1.0, img))
    rgb = np.zeros((patch_h, patch_w, 3), dtype=np.float32)
    rgb[:, :, 0] = np.sqrt(img)
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img ** 2
    return rgb



def load_image():
    while True:
        try:
            year  = 2011 + np.random.randint(4)
            month = 1 + np.random.randint(12)
            day   = 1 + np.random.randint(32)
            hour  = np.random.randint(24)
            minu  = np.random.randint(5)*12
    
            subprocess.call('rm {}/*'.format(work_image_dir),shell=True)
            local_fn =  work_image_dir + '/image.fits'
            cmd = 'aws s3 cp "s3://sdo/aia193/720s/{:04}/{:02}/{:02}/{:02}{:02}.fits" {} --region us-west-2 --quiet'.format(year,month,day,hour,minu, local_fn)
            subprocess.call(cmd, shell=True)
            h = fits.open(local_fn); h[1].verify('fix')
            exptime = h[1].header['EXPTIME']
            if exptime <=0:
                print "EXPTIME <=0"
                continue
            img = intp.zoom(h[1].data.astype(np.float32),zoom=img_w/4096.0,order=0)
            img = scale_brightness(img  / exptime)
            return img
        except Exception,e:
            print e
            continue




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




class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 6*6*512, wscale=0.02*math.sqrt(nz)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0l = L.BatchNormalization(6*6*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 6, 6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x



class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(6*6*512, 2, wscale=0.02*math.sqrt(6*6*512)),
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






def train_dcgan_labeled(gen, dis, epoch0=0):
    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    zvis = (xp.random.uniform(-1, 1, (100, nz), dtype=np.float32))
    
    for epoch in xrange(epoch0,n_epoch):
        perm = np.random.permutation(n_train)
        sum_l_dis = np.float32(0)
        sum_l_gen = np.float32(0)
        
        for i in xrange(0, n_train, batchsize):
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x2 = np.zeros((batchsize, 1, patch_h,patch_w), dtype=np.float32)
            img = load_image()
            for j in range(batchsize):
                rndx = np.random.randint(img_w-patch_w)
                rndy = np.random.randint(img_h-patch_h)

                x2[j,0,:,:] = img[rndx:rndx+patch_w,rndy:rndy+patch_h]
            #print "load image done"
            
            # train generator
            z = Variable(xp.random.uniform(-1, 1, (batchsize, nz), dtype=np.float32))
            x = gen(z)
            yl = dis(x)
            L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
            L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))
            
            # train discriminator
                    
            x2 = Variable(cuda.to_gpu(x2))
            yl2 = dis(x2)
            L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))
            
            #print "forward done"

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()
            
            sum_l_gen += L_gen.data.get()
            sum_l_dis += L_dis.data.get()
            
            #print "backward done"

            if i%image_save_interval==0:
                print "visualize...", epoch, i
                pylab.rcParams['figure.figsize'] = (16.0,16.0)
                pylab.clf()
                vissize = 100
                z = zvis
                z[50:,:] = (xp.random.uniform(-1, 1, (50, nz), dtype=np.float32))
                z = Variable(z)
                x = gen(z, test=True)
                x = x.data.get()
                for i_ in range(100):
                    tmp = data_to_image(i_, x)
                    pylab.subplot(10,10,i_+1)
                    pylab.imshow(tmp)
                    pylab.axis('off')
                pylab.savefig('%s/vis_%d_%06d.png'%(out_image_dir, epoch,i))
                print "visualized."                
        serializers.save_hdf5("%s/dcgan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/dcgan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/dcgan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/dcgan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        print 'epoch end', epoch, sum_l_gen/n_train, sum_l_dis/n_train




gen = Generator()
dis = Discriminator()
gen.to_gpu()
dis.to_gpu()


try:
    os.mkdir(work_image_dir)
    os.mkdir(out_image_dir)
    os.mkdir(out_image_show_dir)
    os.mkdir(out_model_dir)
except:
    pass

train_dcgan_labeled(gen, dis)
