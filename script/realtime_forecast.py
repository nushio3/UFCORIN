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
import population_table as poptbl
import contingency_table

# Parse the command line argument
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

# Obtain MySQL Password
with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()

# Create SQLAlchemy Interface Classes
GOES = goes.GOES
HMI  = wavelet.db_class('hmi.M_720s_nrt', 'haar')

window_minutes = 2**15 # about 22.7 days, which is more than 1/2 of the Sun's rotation period
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
grad_clip = 80.0 #so that exp(grad_clip) < float_max

# Convert the raw GOES and HMI data
# so that they are non-negative numbers of order 1
def encode_goes(x):
    return 10 + math.log(max(1e-10,x))/math.log(10.0)
def decode_goes(x):
    return 10.0**(x-10)

def encode_hmi(x):
    return math.log(max(1.0,x))
def decode_goes(x):
    return math.exp(x)

flare_threshold = {'X': encode_goes(1e-4), '>=M': encode_goes(1e-5),'>=C': encode_goes(1e-6)}
flare_classes = flare_threshold.keys()


# maintain the contingency table.
contingency_tables = dict()
for i in range(n_outputs):
    for c in flare_classes:
        contingency_tables[i,c] = contingency_table.ContingencyTable()
try:
    with open('contingency_tables.pickle','r') as fp:
        contingency_tables = pickle.load(fp,protocol=2)
except:
    pass


# count the populations for each kind of predicted events
poptable = n_outputs * [poptbl.PopulationTable()]
try:
    with open('poptable.pickle','r') as fp:
        poptable = pickle.load(fp,protocol=2)
except:
    pass



# setup the model
model = chainer.FunctionSet(embed=F.Linear(n_inputs, n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, n_outputs))

# Load the model, if available.
try:
    with open('model.pickle','r') as fp:
        model = pickle.load(fp,protocol=2)
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

# flare_x_posi = session.query(sql.func.count('*')).filter(GOES.xray_flux_long >= 1e-6).all()
# flare_x_nega = session.query(sql.func.count('*')).filter(GOES.xray_flux_long <  1e-6).all()
# print flare_x_posi, flare_x_nega

epoch=0
while True:
    # Select the new time range
    d = random.randrange(365*5*24)
    time_begin = datetime.datetime(2011,1,1) +  datetime.timedelta(hours=d)
    time_end   = time_begin +  datetime.timedelta(minutes=window_minutes)
    print time_begin, time_end
    ret_goes = session.query(GOES).filter(GOES.t_tai>=time_begin, GOES.t_tai<=time_end).all()
    if len(ret_goes) < 0.8 * window_minutes :
        print 'too few  GOES data'
        continue
    ret_hmi = session.query(HMI).filter(HMI.t_tai>=time_begin, HMI.t_tai<=time_end).all()
    if len(ret_hmi)  < 0.8 * window_minutes/12 :
        print 'too few HMI data'
        continue

    epoch+=1
    print "epoch=", epoch, len(ret_goes), len(ret_hmi)


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

        # count the population in order to learn from imbalanced number of events.
        for i in range(n_outputs):
            poptable[i].add_event(output_data[i])

        input_variable=chainer.Variable(input_batch)
        output_variable=chainer.Variable(output_batch)

        state, output_prediction = forward_one_step(input_variable, state)

        # accumulate the gradient, modified by the factor 
        fac = []
        for i in range(n_outputs):
            b,a = poptable[i].population_ratio(output_data[i])
            is_overshoot = output_prediction.data[0][i] >= output_data[i]
            fac.append(a if is_overshoot else b)

        fac_variable = np.array([fac], dtype=np.float32)
        loss_iter = F.sum(fac_variable * (output_variable - output_prediction)**2)
        accum_loss += loss_iter

        # collect prediction statistics
        for i in range(n_outputs):
            for c in flare_classes:
                thre = flare_threshold[c]
                p = output_prediction.data[0][i] >= thre
                o = output_variable.data[0][i] >= thre
                contingency_tables[i,c].add(p,o)

        # learn
        if (t+1) % n_backprop == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if t&(t-1)==0:
            print 't=',t,' loss=', loss_iter.data
            for j in [0,4,23]:
                i = j+24
                t = j+1
                print '{}hr:'.format(t),
                for c in flare_classes:
                    print '{} {}'.format(c,contingency_tables[i,c].tss()),
            print

    if True:
            with open('model.pickle','w') as fp:
                pickle.dump(model,fp,protocol=2)
            with open('poptable.pickle','w') as fp:
                pickle.dump(poptable,fp,protocol=2)
            with open('contingency_tables.pickle','w') as fp:
                pickle.dump(contingency_tables,fp,protocol=2)
            print 'dump done'
