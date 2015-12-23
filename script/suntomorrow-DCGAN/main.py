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
parser.add_argument('--norm', default='dcgan',
                    help='Use dcgan/L2 norm.')
parser.add_argument('--epoch0', default=0,type=int,
                    help='set epoch counter')
parser.add_argument('--fresh-start', action='store_true',
                    help='Start simulation anew')
args = parser.parse_args()


nz = 100          # # of dim for final layer
batchsize=25
patch_pixelsize=128
n_epoch=10000
n_train=2000
save_interval =200

n_timeseries = 6

n_movie=120
hardmode_duration= 1


out_image_dir = './out_images_%s'%(args.gpu)
out_model_dir = './out_models_%s'%(args.gpu)


subprocess.call("mkdir -p %s "%(out_image_dir),shell=True)
subprocess.call("mkdir -p %s "%(out_model_dir),shell=True)


################################################################
## Additional Combinators
################################################################

def dual_log(scale, x):
    x2 = x/scale
    #not implemented error!
    #return F.where(x>0,F.log(x2+1),-F.log(1-x2))
    return np.log(1+np.maximum(0,x2))-np.log(1+np.maximum(0,-x2))


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
            c4 = L.Convolution2D(512, 1024, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc4 = L.Deconvolution2D(1024,512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*1024)),
            dc3 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc1 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc0 = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            dc3h = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2h = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc1h = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc0h = L.Deconvolution2D(64, 1, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
            bn4 = L.BatchNormalization(1024),
            bn0d = L.BatchNormalization(64),
            bn1d = L.BatchNormalization(128),
            bn2d = L.BatchNormalization(256),
            bn3d = L.BatchNormalization(512),
            bn0h = L.BatchNormalization(64),
            bn1h = L.BatchNormalization(128),
            bn2h = L.BatchNormalization(256)
        )
        
    def __call__(self, x, test=False):
        h64 = elu(self.c0(x))     # no bn because images from generator will katayotteru?
        h32 = elu(self.bn1(self.c1(h64), test=test))
        h16 = elu(self.bn2(self.c2(h32), test=test))
        h8  = elu(self.bn3(self.c3(h16), test=test))
        h1  = elu(self.bn4(self.c4(h8), test=test))
        
        # idea: not simple addition, but concatenation?
        h = F.relu(self.bn3d(self.dc4(h1), test=test))
        h = F.relu(self.bn2d(self.dc3(h), test=test)) + F.relu(self.bn2h(self.dc3h(h8), test=test))
        h = F.relu(self.bn1d(self.dc2(h), test=test)) + F.relu(self.bn1h(self.dc2h(h16), test=test))
        h = F.relu(self.bn0d(self.dc1(h), test=test)) + F.relu(self.bn0h(self.dc1h(h32), test=test))
        ret=self.dc0(h) + self.dc0h(h64) 
        
        return ret

        #return F.split_axis(x,[1],1)[0]+ret
        #return Variable(xp.zeros((batchsize, 1, 128,128), dtype=np.float32))







class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0i = L.Convolution2D(n_timeseries-1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c0o = L.Convolution2D(1, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            l4l = L.Linear(8*8*512, 2, wscale=0.02*math.sqrt(8*8*512)),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x1,x2, test=False):
        
        x=self.c0i(x1)+self.c0o(x2)
        h = elu(x)     # no bn because images from generator will katayotteru?
        h = elu(self.bn1(self.c1(h), test=test))
        h = elu(self.bn2(self.c2(h), test=test))
        h = elu(self.bn3(self.c3(h), test=test))
        l = self.l4l(h)
        return l





################################################################
## Main program
################################################################


global month,day
month=1#np.random.randint(12)+1
day=0#np.random.randint(31)+1

def load_movie():
    global month,day
    day+=1
    if day>=32:
        month+=1
        day=1
    if month>=13:
        month=1
    # trial mode
    month=np.random.randint(12)+1
    day=np.random.randint(31)+1

    print "now loading... %d/%d"%(month,day)
    subprocess.call('rm aia193/*', shell=True)
    subprocess.call('aws s3 sync "s3://sdo/aia193/720s-x1024/2011/%02d/%02d/" aia193 --region us-west-2 --quiet'%(month,day), shell=True)
    current_movie = n_movie*[None]
    for hr in range(n_movie/5):
        for step in range(5):
            cnt=hr*5+step
            fn = 'aia193/%02d%02d.npz'%(hr,step*12)
            if os.path.exists(fn):
                current_movie[cnt] = dual_log(500,np.load(fn)['img'])
    return current_movie

def create_batch(current_movie_in, current_movie_out):
    for t in range(n_timeseries):
        if current_movie_in[t] is None:
            return None
        if current_movie_out[t] is None:
            return None

    pw=patch_pixelsize
    ph=patch_pixelsize

    ret_in  = np.zeros((batchsize, n_timeseries-1, ph, pw), dtype=np.float32)
    ret_out = np.zeros((batchsize, 1, ph, pw), dtype=np.float32)
    
    for j in range(batchsize):
        oh, ow = current_movie_in[0].shape
        left  = np.random.randint(ow-pw)
        top   = np.random.randint(oh-ph)
        for t in range(n_timeseries):
            if t==n_timeseries -1:
                ret_out[j,0,:,:]=current_movie_out[t][top:top+ph, left:left+pw]
            else:
                ret_in[j,t,:,:]=current_movie_in[t][top:top+ph, left:left+pw]
    return (ret_in, ret_out)


# map a seris of image to image at the next, using the chainer function evol
def evolve_image(evol,imgs):
    h,w=imgs[0].shape
    n = w / patch_pixelsize
    pw=patch_pixelsize
    ph=patch_pixelsize
    #  batch_in = np.zeros((n*n, n_timeseries-1, ph,pw ), dtype=np.float32)
    #  for j in range(n):
    #      for i in range(n):
    #          for t in range(n_timeseries-1):
    #              top=j*ph
    #              left=i*pw
    #              batch_in[j*n+i, t, :, :] = imgs[t][top:top+ph, left:left+pw]
    #  
    #  input_val=Variable(cuda.to_gpu(batch_in))
    #  output_data=evol(input_val).data.get()
    #  
    #  ret = np.zeros((h,w), dtype=np.float32)
    #  for j in range(n):
    #      for i in range(n):
    #          top=j*ph
    #          left=i*pw
    #          ret[top:top+ph, left:left+pw] = output_data[j*n+i, 0, :, :]
    #  return ret

    input_val = Variable(cuda.to_gpu(np.reshape(np.concatenate(imgs), (1,n_timeseries-1,h,w))))
    output_val = evol(input_val,test=False)
    return np.reshape(output_val.data.get(), (h,w))
    

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
        
        for train_ctr in xrange(0, n_train, batchsize):
            print "train:",train_ctr
            # discriminator
            # 0: from dataset
            # 1: from noise

            good_movie=True
            prediction_movie=n_movie*[None]
            current_movie = load_movie()

            vis_kit = {}
            for i in range(n_timeseries-1):
                if current_movie[i] is None:
                    good_movie=False
                else:
                    prediction_movie[i]=current_movie[i]
            if not good_movie: continue
            for i in range(n_timeseries-1,n_movie):
                prediction_movie[i] = evolve_image(evol,prediction_movie[i-n_timeseries+1 : i])
            

            movie_in = None
            movie_out = None
            movie_out_predict=None
            evol_scores = {}
            matsuoka_shuzo = {}
            difficulties = ['normal','hard']
            for difficulty in difficulties:
                evol_scores[difficulty] = [0.0]
                matsuoka_shuzo[difficulty] = True
            for train_offset in range(0,n_movie-n_timeseries):
                for difficulty in difficulties:
                    movie_clip = current_movie[train_offset:train_offset+n_timeseries]
                    if not matsuoka_shuzo[difficulty]:
                        # Doushitesokode yamerunda...
                        continue
                    else:
                        # Akiramen'nayo!
                        pass

                    if difficulty == 'normal':
                        movie_clip_in = movie_clip
                    else:
                        movie_clip_in = prediction_movie[train_offset:train_offset+n_timeseries]
                    maybe_dat = create_batch(movie_clip_in, movie_clip)
                    if not maybe_dat : 
                        print "Warning: skip offset", train_offset, "because of unavailable data."
                        continue
                    data_in, data_out = maybe_dat
                    movie_in =  Variable(cuda.to_gpu(data_in))
                    movie_out = Variable(cuda.to_gpu(data_out))
                    
                    movie_out_predict = evol(movie_in)

                    vis_kit[difficulty] = (movie_in.data.get(),
                                          movie_out.data.get(),
                                          movie_out_predict.data.get())


                    if args.norm == 'dcgan':
                        yl = dis(movie_in,movie_out_predict)
                        L_evol = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
                        L_dis  = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))

                        # train discriminator
                        yl_train = dis(movie_in,movie_out)
                        L_dis += F.softmax_cross_entropy(yl_train, Variable(xp.zeros(batchsize, dtype=np.int32)))

                    else:
                        L2norm = (movie_out - movie_out_predict)**2
                        yl = F.sum(L2norm) / L2norm.data.size
                        L_evol = yl
                    
    
                    evol_scores[difficulty] += [L_evol.data.get()] # np.average(F.softmax(yl).data.get()[:,0])
    
                    
                    
                    o_evol.zero_grads()
                    L_evol.backward()
                    o_evol.update()
                    
                    if args.norm == 'dcgan':
                        o_dis.zero_grads()
                        L_dis.backward()
                        o_dis.update()
                    
                    movie_out_predict.unchain_backward()
                    yl.unchain_backward()
                    L_evol.unchain_backward()
                    if args.norm == 'dcgan':
                        yl_train.unchain_backward()
                        L_dis.unchain_backward()
    
                    sys.stdout.write('%d %6s %s: %f %f\r'%(train_offset,difficulty, args.norm,
                                                           np.average(evol_scores['normal']), np.average(evol_scores['hard'])))
                    sys.stdout.flush()

                    # prevent too much learning from noisy prediction.
                    if len(evol_scores['hard'])>=5 and np.average(evol_scores['hard'][-5:-1]) > 5 * np.average(evol_scores['normal']):
                        matsuoka_shuzo['hard'] = False

            print
            for difficulty in difficulties:
                if vis_kit[difficulty] is None:
                    continue
                movie_data, movie_out_data, movie_pred_data = vis_kit[difficulty]
                imgfn = '%s/batch-%s_%d_%04d.png'%(out_image_dir,difficulty, epoch,train_ctr)
        
                n_col=n_timeseries+2
                plt.rcParams['figure.figsize'] = (1.0*n_col,1.0*batchsize)
                plt.close('all')
        
                for ib in range(batchsize):
                    for j in range(n_timeseries-1):
                        plt.subplot(batchsize,n_col,1 + ib*n_col + j)
                        plt.imshow(movie_data[ib,j,:,:],vmin=0,vmax=1.4)
                        plt.axis('off')
        
                    plt.subplot(batchsize,n_col,1 + ib*n_col + n_timeseries-1)
                    plt.imshow(movie_pred_data[ib,0,:,:],vmin=0,vmax=1.4)
                    plt.axis('off')
        
                    plt.subplot(batchsize,n_col,1 + ib*n_col + n_timeseries+1)
                    plt.imshow(movie_out_data[ib,0,:,:],vmin=0,vmax=1.4)
                    plt.axis('off')
                
                plt.suptitle(imgfn)
                plt.savefig(imgfn)
                subprocess.call("cp %s ~/public_html/suntomorrow-batch-%s-%s.png"%(imgfn,difficulty,args.gpu),shell=True)

            if train_ctr%save_interval==0:
                for answer_mode in ['predict','observe']:
                    for offset in [6,16,32,64,119]:
                        if offset >= n_movie: continue
                        img_prediction = prediction_movie[offset]
                        if answer_mode == 'observe':
                            img_prediction = current_movie[offset]                            
                        if img_prediction is None: continue
                        imgfn = '%s/futuresun_%d_%04d_%s+%03d.png'%(out_image_dir, epoch,train_ctr,answer_mode,offset)
                        plt.rcParams['figure.figsize'] = (12.0, 12.0)
                        plt.close('all')
                        plt.imshow(img_prediction,vmin=0,vmax=1.4)
                        plt.suptitle(imgfn)
                        plt.savefig(imgfn)
                        subprocess.call("cp %s ~/public_html/futuresun/"%(imgfn),shell=True)

                # we don't have enough disk for history
                history_dir = 'history/' #%d-%d'%(epoch,  train_ctr)
                subprocess.call("mkdir -p %s "%(history_dir),shell=True)
                subprocess.call("cp %s/*.h5 %s "%(out_model_dir,history_dir),shell=True)
                
                if epoch==0 and train_ctr==0:
                    continue # no sense to save the initial state.
                print 'saving model...'
                serializers.save_hdf5("%s/dcgan_model_dis.h5"%(out_model_dir),dis)
                serializers.save_hdf5("%s/dcgan_model_evol.h5"%(out_model_dir),evol)
                serializers.save_hdf5("%s/dcgan_state_dis.h5"%(out_model_dir),o_dis)
                serializers.save_hdf5("%s/dcgan_state_evol.h5"%(out_model_dir),o_evol)
                print '...saved.'



xp = cuda.cupy
cuda.get_device(int(args.gpu)).use()

evol = Evolver()
dis = Discriminator()
evol.to_gpu()
dis.to_gpu()

train_dcgan_labeled(evol,dis,epoch0=args.epoch0)
