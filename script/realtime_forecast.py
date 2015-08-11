#!/usr/bin/env python

import argparse, datetime, math, os, pickle, random
import astropy.time as time
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import numpy as np
import sqlalchemy as sql
from   sqlalchemy.orm import sessionmaker
from   sqlalchemy.ext.declarative import declarative_base

import goes.schema as goes
import jsoc.wavelet as wavelet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--optimizer', '-o', default='AdaGrad',
                    help='Name of the optimizer function')
parser.add_argument('--optimizeroptions', '-p', default='()',
                    help='Tuple of options to the optimizer')
parser.add_argument('--filename', '-f', default='',
                    help='Model dump filename tag')
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np

with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()

GOES = goes.GOES
HMI  = wavelet.db_class('hmi.M_720s_nrt', 'haar')

window_minutes = 2**15 # about 22.7 days, which is larger than 1/2 of the Sun's rotation period
hmi_columns = wavelet.subspace_db_columns(2,'S') +  wavelet.subspace_db_columns(2,'NS') 
n_hmi_feature = len(hmi_columns) + 1
n_goes_feature = 3

feature_data = None
target_data = None
n_feature = n_goes_feature + n_hmi_feature

n_backprop = 4096

n_inputs = n_feature
n_outputs = 48
n_units = 720
batchsize = 1
grad_clip = 80.0 #exp(grad_clip) < float_max

# setup the model
model = chainer.FunctionSet(embed=F.Linear(n_inputs, n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, n_outputs))
try:
    with open('model.pickle','r') as fp:
        model = pickle.load(fp)
except:
    print "cannot load model!"


for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)


if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()
# Setup optimizer
optimizer_expr = 'optimizers.{}{}'.format(args.optimizer, args.optimizeroptions)
optimizer = eval(optimizer_expr)
optimizer.setup(model.collect_parameters())

def forward_one_step(x, state, train=True):
    drop_ratio = 0.5
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0,ratio=drop_ratio, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)

    h2_in = model.l2_x(F.dropout(h1,ratio=drop_ratio, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)

    y = model.l3(F.dropout(h2,ratio=drop_ratio, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, y

def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}



def encode_goes(x):
    return 10 + math.log(max(1e-10,x))/math.log(10.0)
def decode_goes(x):
    return 10.0**(x-10)

def encode_hmi(x):
    return math.log(max(1.0,x))
def decode_goes(x):
    return math.exp(x)

goes_range_max_inner_memo = dict()

def goes_range_max_inner(begin, end, stride):
    ret = None
    if begin>=end: return None
    if (begin,end) in goes_range_max_inner_memo: return goes_range_max_inner_memo[(begin,end)]
    if stride <= 1: 
        for i in range(begin,end):
            ret = max(ret,target_data[i])
    else:
        bs = (begin+stride-1) // stride
        es = end // stride
        if es - bs >= 2:
            ret = max(goes_range_max_inner(begin,bs*stride,stride/4), goes_range_max_inner(es*stride,end,stride/4))
            for i in range(es-bs):
                ret = max(ret, goes_range_max_inner((bs+i)*stride,(bs+i+1)*stride,stride/2))
        else:
            ret = goes_range_max_inner(begin,end,stride/2)
    goes_range_max_inner_memo[(begin,end)] = ret
    return ret

def goes_range_max(begin, end):
    if begin < 0: return None
    if end > window_minutes : return None
    return goes_range_max_inner(begin, end , 2**15)


engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))

Session = sessionmaker(bind=engine)
session = Session()


while True:
    d = random.randrange(365*5*24)
    time_begin = datetime.datetime(2011,1,1) +  datetime.timedelta(hours=d)
    time_end   = time_begin +  datetime.timedelta(minutes=window_minutes)
    print time_begin, time_end
    ret_goes = session.query(GOES).filter(GOES.t_tai>=time_begin, GOES.t_tai<=time_end).all()
    if len(ret_goes) < 0.8 * window_minutes :
        print 'short GOES'
        continue
    ret_hmi = session.query(HMI).filter(HMI.t_tai>=time_begin, HMI.t_tai<=time_end).all()
    if len(ret_hmi)  < 0.8 * window_minutes/12 : 
        print 'short HMI'
        continue
    print len(ret_goes), len(ret_hmi)

    # fill the feature matrix
    feature_data = window_minutes * [n_feature * [0.0]]
    target_data = window_minutes * [0.0]
    for row in ret_goes:
        idx = int((row.t_tai - time_begin).total_seconds() / 60)
        feature_data[idx][0] = 1.0
        feature_data[idx][1] = encode_goes(row.xray_flux_long)
        feature_data[idx][2] = encode_goes(row.xray_flux_short)
        target_data[idx]     = encode_goes(row.xray_flux_long)
    for row in ret_hmi:
        idx = int((row.t_tai - time_begin).total_seconds() / 60)
        o = n_goes_feature
        feature_data[idx][o] = 1.0
        for j in range(len(hmi_columns)):
            col_str = hmi_columns[j]
            feature_data[idx][o+1+j] = encode_hmi(getattr(row,col_str))
    print 'feature filled.'

    while False: # Test code for goes_range_max
        b = random.randrange(window_minutes)
        e = random.randrange(window_minutes)
        if not b<e : continue
        print goes_range_max(b,e), max(target_data[b:e])

    # start BPTT learning anew
    state = make_initial_state()

    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    n_backprop = int(2**random.randrange(1,10))
    print 'backprop length = ', n_backprop
    for t in range(window_minutes - 24*60): # future max prediction training
        input_batch = np.array([feature_data[t]], dtype=np.float32)
        output_data = []
        for i in range(24):
            output_data.append(goes_range_max(t+60*i,t+60*(i+1)))
        for i in range(24):
            output_data.append(goes_range_max(t,t+60*(i+1)))
        output_batch = np.array([output_data], dtype=np.float32)
        
        input_variable=chainer.Variable(input_batch)
        output_variable=chainer.Variable(output_batch)
        
        state, output_prediction = forward_one_step(input_variable, state)
        
        loss_iter = F.mean_squared_error(output_variable, output_prediction)
        accum_loss += loss_iter
        if t&(t-1)==0:
            print 't=',t,' loss=', loss_iter.data
        if (t+1) % n_backprop == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()
    with open('model.pickle','w') as fp:
        pickle.dump(model,fp)





            
