#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, datetime, math, os, pickle, random
import astropy.time as time
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sqlalchemy as sql
from   sqlalchemy.orm import sessionmaker
from   sqlalchemy.ext.declarative import declarative_base
import subprocess

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
parser.add_argument('--optimizeroptions', '-p', default='(lr=0.001)',
                    help='Tuple of options to the optimizer')
parser.add_argument('--filename', '-f', default='',
                    help='Model dump filename tag')
parser.add_argument('--realtime', '-r', default='',
                    help='Perform realtime prediction')
args = parser.parse_args()
mod = cuda if args.gpu >= 0 else np




# Obtain MySQL Password
with open(os.path.expanduser('~')+'/.mysqlpass','r') as fp:
    password = fp.read().strip()

# Create SQLAlchemy Interface Classes
GOES = goes.GOES
HMI  = wavelet.db_class('hmi.M_720s_nrt', 'haar')

dt = datetime.timedelta(seconds=720.0)
t_per_hour = int(round(datetime.timedelta(hours=1).total_seconds() / dt.total_seconds()))

window_size = 2**11 # about 17 days, which is more than 1/2 of the Sun's rotation period
hmi_columns = wavelet.subspace_db_columns(2,'S') +  wavelet.subspace_db_columns(2,'NS')
n_hmi_feature = len(hmi_columns) + 1
n_goes_feature = 3

feature_data = None
target_data = None
n_feature = n_goes_feature + n_hmi_feature


n_inputs = n_feature
n_outputs = 48
n_units = 720
batchsize = 1
grad_clip = 40.0 #so that exp(grad_clip) < float_max

# Convert the raw GOES and HMI data
# so that they are non-negative numbers of order 1
def encode_goes(x):
    return 10 + math.log(max(1e-10,x))/math.log(10.0)
def decode_goes(x):
    return 10.0**(x-10)

def encode_hmi(x):
    return math.log(max(1.0,x))
def decode_hmi(x):
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
        contingency_tables = pickle.load(fp)
except:
    print "cannot load contingency table!"


# count the populations for each kind of predicted events
poptable = n_outputs * [poptbl.PopulationTable()]
try:
    with open('poptable.pickle','r') as fp:
        poptable = pickle.load(fp)
except:
    print "cannot load poptable!"



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
    return goes_range_max_inner(max(0,begin), min(window_size,end) , 2**15)


engine = sql.create_engine('mysql+mysqldb://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))

Session = sessionmaker(bind=engine)
session = Session()



epoch=0
while True:
    for i in range(n_outputs):
        for c in flare_classes:
            contingency_tables[i,c].attenuate(1e-3)

    # Select the new time range
    d = random.randrange(365*5*24)
    time_begin = datetime.datetime(2011,1,1) +  datetime.timedelta(hours=d)

    if args.realtime:
        now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime
        time_begin = now - (window_size -  24*t_per_hour) * dt

    time_end   = time_begin + window_size * dt
    print time_begin, time_end,
    ret_goes = session.query(GOES).filter(GOES.t_tai>=time_begin, GOES.t_tai<=time_end).all()
    goes_fill_ratio = len(ret_goes) / (window_size * 12.0)
    if goes_fill_ratio < 0.8 and not args.realtime:
        print 'too few GOES data'
        continue
    ret_hmi = session.query(HMI).filter(HMI.t_tai>=time_begin, HMI.t_tai<=time_end).all()
    hmi_fill_ratio = len(ret_hmi) / (window_size * 1.0)
    if hmi_fill_ratio < 0.8 and not args.realtime:
        print 'too few HMI data'
        continue

    epoch+=1
    nowmsg=datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    print "epoch={} {}({:4.2f}%) {}({:4.2f}%)".format(epoch, len(ret_goes), goes_fill_ratio*100, len(ret_hmi), hmi_fill_ratio*100)
    print "WCT=[{}]".format(nowmsg)



    # fill the feature matrix
    feature_data = np.array(window_size * [n_feature * [0.0]], dtype=np.float32)
    target_data = np.array(window_size * [0.0], dtype=np.float32)
    for row in ret_goes:
        idx = min(window_size-1,int((row.t_tai - time_begin).total_seconds() / dt.total_seconds()))
        feature_data[idx, 0] = 1.0
        feature_data[idx, 1] = max(feature_data[idx, 1], encode_goes(row.xray_flux_long))
        feature_data[idx, 2] = max(feature_data[idx, 2], encode_goes(row.xray_flux_short))
        target_data[idx]     = feature_data[idx, 1]
    for row in ret_hmi:
        idx = min(window_size-1,int((row.t_tai - time_begin).total_seconds() / dt.total_seconds()))
        o = n_goes_feature
        feature_data[idx, o] = 1.0
        for j in range(len(hmi_columns)):
            col_str = hmi_columns[j]
            feature_data[idx, o+1+j] = encode_hmi(getattr(row,col_str))

    print 'feature filled.'

    while False: # Test code for goes_range_max
        b = random.randrange(window_size)
        e = random.randrange(window_size)
        if not b<e : continue
        print goes_range_max(b,e), max(target_data[b:e])

    # start BPTT learning anew
    state = make_initial_state()

    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    n_backprop = 1024 #int(2**random.randrange(1,5))
    print 'backprop length = ', n_backprop

    last_t = window_size - 24*t_per_hour - 1
    for t in range(last_t+1): # future max prediction training
        input_batch = np.array([feature_data[t]], dtype=np.float32)
        output_data = []
        for i in range(24):
            output_data.append(goes_range_max(t+t_per_hour*i,t+t_per_hour*(i+1)))
        for i in range(24):
            output_data.append(goes_range_max(t,t+t_per_hour*(i+1)))
        output_batch = np.array([output_data], dtype=np.float32)

        # count the population in order to learn from imbalanced number of events.
        for i in range(n_outputs):
            poptable[i].add_event(output_data[i])

        input_variable=chainer.Variable(input_batch)
        output_variable=chainer.Variable(output_batch)

        state, output_prediction = forward_one_step(input_variable, state, train = not args.realtime)

        # accumulate the gradient, modified by the factor
        fac = []
        for i in range(n_outputs):
            b,a = poptable[i].population_ratio(output_data[i])
            is_overshoot = output_prediction.data[0, i] >= output_data[i]
            if output_data[i] == encode_goes(0) or output_data[i] == None:
                factor=0
            else:
                factor=1.0 #a if is_overshoot else b
            fac.append(factor)

        fac_variable = np.array([fac], dtype=np.float32)
        loss_iter = F.sum(fac_variable * (output_variable - output_prediction)**2)/float(len(fac))
        accum_loss += loss_iter

        # collect prediction statistics
        for i in range(n_outputs):
            for c in flare_classes:
                thre = flare_threshold[c]
                p = output_prediction.data[0, i] >= thre
                o = output_variable.data[0, i] >= thre
                contingency_tables[i,c].add(p,o)

        # learn
        if (t+1) % n_backprop == 0 or t==last_t:
            print "backprop.."
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (t&(t-1)==0 or t%1024==0) and t>0 and t%64==0:
            print 't=',t,' loss=', loss_iter.data
            for j in [0,4,23]:
                i = j+24
                pred_len = j+1
                print '{}hr:'.format(pred_len),
                for c in flare_classes:
                    print '{} {}'.format(c,contingency_tables[i,c].tss()),
            print
        if args.realtime == 'quick': break

    if not args.realtime: # at the end of the loop
        print 'dumping...',
        with open('model.pickle','w') as fp:
            pickle.dump(model,fp,protocol=-1)
        with open('poptable.pickle','w') as fp:
            pickle.dump(poptable,fp,protocol=-1)
        with open('contingency_tables.pickle','w') as fp:
            pickle.dump(contingency_tables,fp,protocol=-1)
        print 'dump done'
    if args.realtime:
        # visualize forecast
        fig, ax = plt.subplots()
        ax.set_yscale('log')

        goes_curve_t = [time_begin + i*dt for i in range(window_size)]
        goes_curve_y = [decode_goes(target_data[i]) if target_data[i] != encode_goes(0) else None for i in range(window_size)]
        ax.plot(goes_curve_t, goes_curve_y, 'b')

        pred_data = output_prediction.data[0]
        pred_curve_t = []
        pred_curve_y = []
        for i in range(24):
            pred_begin_t = now + t_per_hour*i*dt
            pred_end_t   = now + t_per_hour*(i+1)*dt
            pred_flux = decode_goes(pred_data[i])
            pred_curve_t += [pred_begin_t, pred_end_t]
            pred_curve_y += [pred_flux,pred_flux]
        ax.plot(pred_curve_t, pred_curve_y, 'g')
        for i in range(24):
            pred_begin_t = now
            pred_end_t   = now + t_per_hour*(i+1)*dt
            pred_flux = decode_goes(pred_data[i+24])
            pred_curve_t = [pred_begin_t, pred_end_t]
            pred_curve_y = [pred_flux,pred_flux]
            ax.plot(pred_curve_t, pred_curve_y, 'r')

        days    = mdates.DayLocator()  # every day
        daysFmt = mdates.DateFormatter('%Y-%m-%d')
        hours   = mdates.HourLocator()
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFmt)
        ax.xaxis.set_minor_locator(hours)
        ax.grid()
        fig.autofmt_xdate()
        ax.set_title('GOES Forecast produced at {}(TAI)'.format(now.strftime('%Y-%m-%d %H:%M:%S')))
        ax.set_xlabel('International Atomic Time')
        ax.set_ylabel(u'GOES Long[1-8Å] Xray Flux')

        plt.savefig('prediction-result.png', dpi=200)
        subprocess.call('cp prediction-result.png ~/public_html/', shell=True)
        plt.close('all')

        exit(0)
