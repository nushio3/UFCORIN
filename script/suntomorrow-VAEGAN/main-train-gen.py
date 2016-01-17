#!/usr/bin/env python
# http://arxiv.org/pdf/1512.09300.pdf

import pickle,subprocess, argparse, urllib
from astropy.io import fits
import scipy.ndimage.interpolation as intp

import numpy as np
import os,re
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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





parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID')
parser.add_argument('--batchsize', default=2, type=int,
                    help='how many batches to train simultaneously.')
parser.add_argument('--gamma', default=1.0,type=float,
                    help='weight of content similarity over style similarity')
parser.add_argument('--creativity-weight', default=1.0,type=float,
                    help='weight of creativity over emulation')
parser.add_argument('--stride','-s', default=4,type=int,
                    help='stride size of the final layer')
parser.add_argument('--final-filter-size', default=8,type=int,
                    help='size of the final filter')
parser.add_argument('--nz', default=100,type=int,
                    help='the size of encoding space')
parser.add_argument('--dropout', action='store_true',
                    help='use dropout when training dis.')
parser.add_argument('--shake-camera', action='store_true',
                    help='shake camera to prevent overlearning.')
parser.add_argument('--enc-norm', default = 'dis',
                    help='use (dis/L2) norm to train encoder.')
parser.add_argument('--normalization', default = 'batch',
                    help='use (batch/channel) normalization.')
parser.add_argument('--prior-distribution', default = 'gaussian',
                    help='use (uniform/gaussian) distribution for z prior.')
parser.add_argument('--Phase', default = 'gen',
                    help='train (gen/enc/evol).')

args = parser.parse_args()

xp = cuda.cupy
cuda.get_device(args.gpu).use()

def foldername(args):
    x = urllib.quote(str(args))
    x = re.sub('%..','_',x)
    x = re.sub('___','-',x)
    x = re.sub('Namespace_','',x)
    return x

work_image_dir = '/mnt/work-{}'.format(args.gpu)
out_image_dir = '/mnt/public_html/out-images-{}'.format(foldername(args))
out_image_show_dir = '/mnt/public_html/out-images-{}'.format(args.gpu)
out_model_dir = './out-models-{}'.format(args.gpu)


img_w=512 # size of the image
img_h=512
nz = args.nz          # # of dim for Z
n_signal = 2 # # of signal
zw = (img_w/16-args.final_filter_size) / args.stride +1 # size of in-vivo z patch
zh = zw


n_epoch=10000
n_train=10000
image_save_interval = 100


def average(x):
    return F.sum(x/x.data.size)

# A scaling for human perception of SDO-AIA 193 image.
# c.f. page 11 of
# http://helio.cfa.harvard.edu/trace/SSXG/ynsu/Ji/sdo_primer_V1.1.pdf
#
# AIA orthodox color table found at
# https://darts.jaxa.jp/pub/ssw/sdo/aia/idl/pubrel/aia_lct.pro

def scale_brightness(x):
    lo = 50.0
    hi = 1250.0
    x2 = np.minimum(hi, np.maximum(lo,x))
    x3 = (np.log(x2)-np.log(lo)) / (np.log(hi) - np.log(lo))
    return x3

def variable_to_image(var):
    img = var.data.get()[0,0]
    img = np.maximum(0.0, np.minimum(1.0, img))
    rgb = np.zeros((img_h, img_w, 3), dtype=np.float32)
    rgb[:, :, 0] = np.sqrt(img)
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img ** 2
    return rgb



class ELU(function.Function):

    """Exponential Linear Unit."""
    # https://github.com/muupan/chainer-elu

    def __init__(self, alpha=1.0):
        self.alpha = np.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == np.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (np.exp(y[neg_indices]) - 1)
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
        gx[neg_indices] *= self.alpha * np.exp(x[0][neg_indices])
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


def channel_normalize(x, test=False):
    s0,s1,s2,s3 = x.data.shape
    cavg = F.reshape(F.sum(x, axis=1) / s1, (s0,1,s2,s3))
    xavg = F.concat(s1 * [cavg])
    cvar = F.reshape(F.sum((x - xavg)**2, axis=1) / s1, (s0,1,s2,s3))
    xvar = F.concat(s1 * [cvar])    
    return (x - xavg) / (xvar + 1e-5)**0.5

def shake_camera(img):
    if not args.shake_camera:
        return img
    s0,s1,s2,s3 = img.data.shape
    zerobar = Variable(xp.zeros((s0,s1,4,s3),dtype=np.float32))
    img = F.concat([zerobar, img, zerobar],axis=2)
    randshift=np.random.randint(1,8)
    img = F.split_axis(img, [randshift,randshift+img_w],axis=2)[1]

    zerobar = Variable(xp.zeros((s0,s1,s2,4,1),dtype=np.float32))
    img = F.reshape(img,(s0,s1,s2,s3,1))
    img = F.concat([zerobar, img, zerobar],axis=3)
    randshift=np.random.randint(1,8)
    img = F.split_axis(img, [randshift,randshift+img_w],axis=3)[1]
    img = F.reshape(img,(s0,s1,s2,s3))
     
    return img



class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            dc0z = L.Deconvolution2D(nz, 512, args.final_filter_size, stride=args.stride, wscale=0.02*math.sqrt(nz)),
            dc0s = L.Deconvolution2D(n_signal, 512,  args.final_filter_size, stride=args.stride, wscale=0.02*math.sqrt(n_signal)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, z_signal, test=False):
        # h  = F.relu(channel_normalize(self.dc0z(z) + self.dc0s(z_signal), test=test))
        # h = F.relu(channel_normalize(self.dc1(h), test=test))
        # h = F.relu(channel_normalize(self.dc2(h), test=test))
        # h = F.relu(channel_normalize(self.dc3(h), test=test))
        # x = (self.dc4(h))
        # return x
        h  = F.relu(self.bn0(self.dc0z(z) + self.dc0s(z_signal), test=test))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x

class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__(
            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            cz = L.Convolution2D(512, nz , args.final_filter_size, stride=args.stride, wscale=0.02*math.sqrt(8*8*512)),
            
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.relu(self.c0(x))     # no bn because images from generator will katayotteru?
        h = F.relu(self.bn1(self.c1(h), test=test))
        h = F.relu(self.bn2(self.c2(h), test=test))
        h = F.relu(self.bn3(self.c3(h), test=test))
        return self.cz(h)

global coord_image
coord_image = np.zeros((args.batchsize,1,img_h, img_w), dtype=np.float32)

for iy in range(img_h):
    for ix in range(img_w):
        x = 2*float(ix - img_w/2)/img_w
        y = 2*float(iy - img_h/2)/img_h
        coord_image[:,0,iy,ix] = x**2 + y**2
x_signal=Variable(cuda.to_gpu(coord_image))


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*1)),
            c0s= L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*1)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*32)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            cz = L.Convolution2D(512, 2, args.final_filter_size, stride=args.stride,wscale=0.02*math.sqrt(8*8*512)),

            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False, compare=None):
        if compare is not None:
            h =  elu(self.c0(x) + self.c0s(x_signal))  
            h =  elu(channel_normalize(self.c1(h), test=test))
            h =  channel_normalize(self.c2(h), test=test)
            h2 = elu(self.c0(compare) + self.c0s(x_signal))            
            h2 = elu(channel_normalize(self.c1(h2), test=test))
            h2 = channel_normalize(self.c2(h2), test=test)
            
            return average((h-h2)**2)

        h = elu(self.c0(x) + self.c0s(x_signal))     # no bn because images from generator will katayotteru?
        #h = elu(channel_normalize(self.c1(h), test=test))
        #h = elu(channel_normalize(self.c2(F.dropout(h)), test=test))
        #h = elu(channel_normalize(self.c3(F.dropout(h)), test=test))
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(F.dropout(h,train = args.dropout)), test=test))
        h = elu(self.bn3(self.c3(F.dropout(h,train = args.dropout)), test=test))
        h=self.cz(F.dropout(h,train = args.dropout))
        l = F.sum(h,axis=(2,3))/(h.data.size / 2)
        return l



def load_image():
    ret = np.zeros((args.batchsize,1,img_h,img_w),dtype=np.float32)
    i=0
    while i<args.batchsize:
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
            ret[i, :, :, :] = np.reshape(img, (1,1,img_h,img_w))
            i += 1
        except:
            continue
    return ret

def position_signal(i,w):
    ww = w/2
    return (i - ww)/float(ww)


def train_vaegan_labeled(gen, enc, dis, epoch0=0):

    o_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_enc = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    o_gen.setup(gen)
    o_enc.setup(enc)
    o_dis.setup(dis)
    o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_enc.add_hook(chainer.optimizer.WeightDecay(0.00001))
    o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    gamma_p = 1.0
    
    for epoch in xrange(epoch0,n_epoch):
        
        for i in xrange(0, n_train, args.batchsize):
            print (epoch,i),
            # discriminator
            # 0: from dataset
            # 1: from noise

            #print "load image start ", i
            x_train_data = load_image()
            x_train = Variable(cuda.to_gpu(x_train_data))
            

            # generate prior and signal
            if args.prior_distribution == 'uniform':
                z_prior = np.random.uniform(-1,1,(args.batchsize, nz, zh, zw)).astype(np.float32)
            else:
                z_prior = np.random.standard_normal((args.batchsize, nz, zh, zw)).astype(np.float32)
            z_signal =np.zeros((args.batchsize, 2, zh, zw)).astype(np.float32)
            # embed the position signal in z vector
            for y in range (zh):
                for x in range (zw):
                    z_signal[:,0,y,x] = position_signal(x, zw)
                    z_signal[:,1,y,x] = position_signal(y, zh)
            z_prior = Variable(cuda.to_gpu(z_prior))
            z_signal = Variable(cuda.to_gpu(z_signal))


            x_creative = shake_camera(gen(z_prior,z_signal))
            x_train = shake_camera(x_train)

            yl_train  = dis(x_train)
            yl_prior  = dis(x_creative)

            if args.Phase != 'gen':
                # use encoder
                z_enc = enc(x_train)
                x_vae = shake_camera(gen(z_enc,z_signal))
                yl_vae    = dis(x_vae)
                yl_dislike = dis(x_vae, compare=x_train)            
                yl_L2like = average((x_vae - x_train)**2)

            l_prior0 = average(z_prior**2)
            if args.Phase != 'gen':
                l_prior  = average(z_enc**2)

                if float(l_prior.data.get()) < float(l_prior0.data.get()): gamma_p = 0.0
                if float(l_prior.data.get()) > float(l_prior0.data.get()): gamma_p = 1.0

            train_is_genuine = F.softmax_cross_entropy(yl_train, Variable(xp.zeros(args.batchsize, dtype=np.int32)))
            train_is_fake = F.softmax_cross_entropy(yl_train, Variable(xp.ones(args.batchsize, dtype=np.int32)))

            prior_is_genuine= F.softmax_cross_entropy(yl_prior, Variable(xp.zeros(args.batchsize, dtype=np.int32)))
            prior_is_fake = F.softmax_cross_entropy(yl_prior, Variable(xp.ones(args.batchsize, dtype=np.int32)))

            if args.Phase != 'gen':
                vae_is_genuine   = F.softmax_cross_entropy(yl_vae, Variable(xp.zeros(args.batchsize, dtype=np.int32)))
                vae_is_fake   = F.softmax_cross_entropy(yl_vae, Variable(xp.ones(args.batchsize, dtype=np.int32)))
            
            
            if args.Phase == 'gen':
                L_gen  = args.creativity_weight * prior_is_genuine 

                L_dis  = train_is_genuine + prior_is_fake

            else:
                L_gen  = args.creativity_weight * prior_is_genuine + vae_is_genuine + args.gamma * yl_dislike

                L_enc  = vae_is_genuine + gamma_p * l_prior + (yl_L2like if args.enc_norm == 'L2' else yl_dislike)

                L_dis  = 2*train_is_genuine + vae_is_fake + prior_is_fake
            
            for x in ['yl_train', 'yl_vae', 'yl_prior', 'yl_dislike', 'yl_L2like','l_prior','l_prior0','gamma_p','train_is_genuine', 'train_is_fake', 'vae_is_genuine', 'vae_is_fake', 'prior_is_genuine', 'prior_is_fake', 'L_gen', 'L_enc', 'L_dis']:
                print x+":",
                try:
                    vx = eval(x).data.get()
                    if vx.size==1:
                        print float(vx),
                    else:
                        print vx,' ',
                except AttributeError:
                    print eval(x),
                except:
                    pass


            print

            o_gen.zero_grads()
            L_gen.backward()
            o_gen.update()

            if args.Phase != 'gen':            
                o_enc.zero_grads()
                L_enc.backward()
                o_enc.update()
            
            o_dis.zero_grads()
            L_dis.backward()
            o_dis.update()
            
            L_gen.unchain_backward()
            L_dis.unchain_backward()

            if args.Phase != 'gen':
                L_enc.unchain_backward()
            
            #print "backward done"

                

            if i%image_save_interval==0:
                vis_wscale = 2 if args.Phase == 'gen' else 3

                fn0 = '%s/tmp.png'%(out_image_show_dir)
                fn2 = '%s/latest.png'%(out_image_show_dir)
                fn1 = '%s/vis_%02d_%06d.png'%(out_image_dir, epoch,i)

                plt.rcParams['figure.figsize'] = (6.0 * vis_wscale,6.0)
                plt.clf()
                plt.subplot(1,vis_wscale,1)
                plt.imshow(variable_to_image(x_train))
                if args.Phase != 'gen':
                    plt.subplot(1,vis_wscale,2)
                    plt.imshow(variable_to_image(x_vae))
                plt.subplot(1,vis_wscale,vis_wscale)
                plt.imshow(variable_to_image(x_creative))
                plt.suptitle(str(args)+"\n"+'epoch{}-{}'.format(epoch,i))

                plt.savefig(fn0)
                subprocess.call("cp {} {}".format(fn0,fn2), shell=True)
                subprocess.call("cp {} {}".format(fn0,fn1), shell=True)
                
        serializers.save_hdf5("%s/vaegan_model_dis_%d.h5"%(out_model_dir, epoch),dis)
        serializers.save_hdf5("%s/vaegan_state_dis_%d.h5"%(out_model_dir, epoch),o_dis)
        serializers.save_hdf5("%s/vaegan_model_gen_%d.h5"%(out_model_dir, epoch),gen)
        serializers.save_hdf5("%s/vaegan_state_gen_%d.h5"%(out_model_dir, epoch),o_gen)
        serializers.save_hdf5("%s/vaegan_model_enc_%d.h5"%(out_model_dir, epoch),enc)
        serializers.save_hdf5("%s/vaegan_state_enc_%d.h5"%(out_model_dir, epoch),o_enc)
        print('epoch end', epoch)




gen = Generator()
enc = Encoder()
dis = Discriminator()
gen.to_gpu()
enc.to_gpu()
dis.to_gpu()


try:
    subprocess.call('mkdir -p ' + work_image_dir, shell=True)
    subprocess.call('mkdir -p ' + out_image_dir, shell=True)
    subprocess.call('mkdir -p ' + out_image_show_dir, shell=True)
    subprocess.call('mkdir -p ' + out_model_dir, shell=True)
except:
    pass

train_vaegan_labeled(gen, enc, dis)
