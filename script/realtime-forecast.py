#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, copy, datetime, math, os, pickle, random,sys
import astropy.time as time
import chainer
# from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import hashlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sqlalchemy as sql
from   sqlalchemy.orm import sessionmaker
from   sqlalchemy.ext.declarative import declarative_base
import subprocess
import tabulate

import goes.schema as goes
import jsoc.wavelet as wavelet
import population_table as poptbl
import contingency_table

# Parse the command line argument

GPU_STRIDE=2

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID - {}(smaller value indicates CPU)'.format(GPU_STRIDE))
parser.add_argument('--optimizer', '-o', default='AdaGrad',
                    help='Name of the optimizer function')
parser.add_argument('--optimizeroptions', '-p', default='()',
                    help='Tuple of options to the optimizer')
parser.add_argument('--filename', '-f', default='',
                    help='Model dump filename tag')
parser.add_argument('--realtime', '-r', default='',
                    help='Perform realtime prediction')
parser.add_argument('--learn-interval',  default='random',
                    help='learning interval, in hours')
parser.add_argument('--quiet-log', '-q', action='store_true',
                    help='redirect standard output to log file and be quiet')
parser.add_argument('--grad-factor', default='ab',
                    help='gradient priority factor (ab/flat/severe)')
parser.add_argument('--backprop-length',default='accel',
                    help='gradually increase backprop length (accel) or let it be constant (number)')
parser.add_argument('--work-dir', default='',
                    help='working directory')


args = parser.parse_args()
mod = cuda if args.gpu >= GPU_STRIDE else np

workdir='result/' + hashlib.sha256("salt{}{}".format(args,random.random())).hexdigest()
if args.work_dir != '':
    workdir = args.work_dir
subprocess.call('mkdir -p ' + workdir ,shell=True)
subprocess.call('mkdir -p ccmc' ,shell=True)
os.chdir(workdir)

if args.quiet_log:
    sys.stdout=open("stdout.txt","w")

with open("args.log","w") as fp:
    fp.write('{}\n'.format(args))

def to_PU(x):
    if args.gpu>= GPU_STRIDE:
        return cuda.to_gpu(x)
    else:
        return x
def from_PU(x):
    if args.gpu>= GPU_STRIDE:
        return cuda.to_cpu(x)
    else:
        return x

class Forecast:
    def generate_ccmc_submission(self):
        now = datetime.datetime.now()
        filename = "ccmc/UFCORIN_1_{}.txt".format(now.strftime("%Y%m%d_%H%M"))

        ys_log = [math.log10(y[0]) for y in self.pred_max_y]
        pred_mean = ys_log[23]
        pred_stddev = math.sqrt(np.mean([(ys_log[i] - pred_mean)**2 for i in [20,21,22]]))
        def cdf(y):
            return 0.5 * (1 + math.erf((y-pred_mean)/(math.sqrt(2.0) * pred_stddev)))
        prob_x = 1 - cdf(-4)
        prob_m = 1 - cdf(-5)
        prob_c = 1 - cdf(-6)
        time_begin = self.pred_max_t[23][0]
        time_end   = self.pred_max_t[23][1]

        def fmttime(t):
            return t.strftime("%Y-%m-%dT%H:%MZ")

        with open(filename, 'w') as fp:
            fp.write("Forecasting method: UFCORIN_1\n")
            fp.write("Issue Time: {}\n".format(fmttime(now)))
            fp.write("Prediction Window Start Time: {}\n".format(fmttime(time_begin)))
            fp.write("Prediction Window End Time: {}\n".format(fmttime(time_end)))
            fp.write("Probability Bins: M+\n")
            fp.write("Input data: SDO/HMI LOS_Magnetogram, GOES X-ray flux\n")
            spacer = "---- ---- ----"
            fp.write("{:.4f} {} {:.4f} {} {:.4f} {}\n".format(prob_x, spacer, prob_m, spacer,prob_c,spacer))
    def visualize(self, filename):
        now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime
        fig, ax = plt.subplots()
        ax.set_yscale('log')

        ax.plot(self.goes_curve_t, self.goes_curve_y, 'b')

        ax.plot(self.pred_curve_t, self.pred_curve_y, 'g')
        for i in range(24):
            ax.plot(self.pred_max_t[i], self.pred_max_y[i], 'r')

        days    = mdates.DayLocator()  # every day
        daysFmt = mdates.DateFormatter('%Y-%m-%d')
        hours   = mdates.HourLocator()
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(daysFmt)
        ax.xaxis.set_minor_locator(hours)
        ax.grid()
        fig.autofmt_xdate()
        ax.set_title('GOES Forecast at {}(TAI)'.format(now.strftime('%Y-%m-%d %H:%M:%S')))
        ax.set_xlabel('International Atomic Time')
        ax.set_ylabel(u'GOES Long[1-8Å] Xray Flux')

        plt.savefig(filename, dpi=200)
        plt.close('all')

        with open("prediction-result.md","r") as fp:
            md_template=fp.read()

        predicted_goes_flux = self.pred_max_y[-1][-1]
        predicted_class = "Quiet"
        if predicted_goes_flux >= 1e-6: predicted_class = "C Class"
        if predicted_goes_flux >= 1e-5: predicted_class = "M Class"
        if predicted_goes_flux >= 1e-4: predicted_class = "X Class"
        md_new = md_template.replace('{{GOES_FLUX}}','{:0.2}'.format(predicted_goes_flux)).replace('{{FLARE_CLASS}}',predicted_class)
        with open("prediction-result-filled.md","w") as fp:
            fp.write(md_new)
        subprocess.call('pandoc prediction-result-filled.md -o ~/public_html/prediction-result.html'  , shell=True)

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

global feature_data, target_data

feature_data = None
target_data = None
n_feature = n_goes_feature + n_hmi_feature


n_inputs = n_feature
n_outputs = 48
n_units = 720
batchsize = 1
grad_clip = 5.0 #so that exp(grad_clip) < float_max

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

# format the contingency table for monitoring.
def ppr_contingency_table(tbl):
    i = n_outputs - 1
    row0 = []
    row1 = []
    row2 = []
    for c in flare_classes:
        row0 += [c, 'O+', 'O-']
        row1 += ['P+', tbl[i,c].counter[True,True],tbl[i,c].counter[True,False]]
        row2 += ['P-', tbl[i,c].counter[False,True],tbl[i,c].counter[False,False]]
    return tabulate.tabulate([row0,row1,row2],headers='firstrow')



# count the populations for each kind of predicted events
poptable = n_outputs * [poptbl.PopulationTable()]
try:
    with open('poptable.pickle','r') as fp:
        poptable = pickle.load(fp)
except:
    print "cannot load poptable!"

# track the predicted and observed values.
global prediction_trace
prediction_trace = []


# setup the model
model = chainer.FunctionSet(embed=F.Linear(n_inputs, n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
#                            l3a=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, n_outputs))

# Load the model, if available.
try:
    with open('model.pickle','r') as fp:
        model = pickle.load(fp)
except:
    print "cannot load model!"
    for param in model.parameters:
        param[:] = np.random.uniform(-0.1, 0.1, param.shape)

if args.gpu >= GPU_STRIDE:
    cuda.get_device(args.gpu- GPU_STRIDE).use()
    model.to_gpu()

# Setup optimizer
optimizer_expr = 'optimizers.{}{}'.format(args.optimizer, args.optimizeroptions)
sys.stderr.write(optimizer_expr+"\n")
optimizer = eval(optimizer_expr)
optimizer.setup(model)

def forward_one_step(x, state, train=True):
    drop_ratio = 0.5
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0,ratio=drop_ratio, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)

    h2_in = model.l2_x(F.dropout(h1,ratio=drop_ratio, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)

#    ya = F.relu(model.l3a(F.dropout(h2,ratio=drop_ratio, train=train)))
    y =  model.l3(F.dropout(h2,ratio=drop_ratio, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
    return state, y

def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
                                             dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}



global goes_range_max_inner_memo
goes_range_max_inner_memo = dict()
def goes_range_max_inner(begin, end, stride):
    global feature_data, target_data

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


engine = sql.create_engine('mysql+pymysql://ufcoroot:{}@sun-feature-db.cvxxbx1dlxvk.us-west-2.rds.amazonaws.com:3306/sun_feature'.format(password))

Session = sessionmaker(bind=engine)
session = Session()

global epoch
epoch=0

# while True:
#     for i in range(n_outputs):
#         for c in flare_classes:
#             contingency_tables[i,c].attenuate(1e-2)

def learn_predict_from_time(timedelta_hours):
    global feature_data, target_data, prediction_trace
    global epoch
    # Select the new time range
    time_begin = datetime.datetime(2011,1,1) +  datetime.timedelta(hours=timedelta_hours)

    now = time.Time(datetime.datetime.now(),format='datetime',scale='utc').tai.datetime
    if args.realtime:
        time_begin = now - (window_size -  24*t_per_hour) * dt

    time_end   = time_begin + window_size * dt
    print time_begin, time_end,
    ret_goes = session.query(GOES).filter(GOES.t_tai>=time_begin, GOES.t_tai<=time_end).all()
    goes_fill_ratio = len(ret_goes) / (window_size * 12.0)
    if goes_fill_ratio < 0.8 and not args.realtime:
        print 'too few GOES data'
        return
    ret_hmi = session.query(HMI).filter(HMI.t_tai>=time_begin, HMI.t_tai<=time_end).all()
    hmi_fill_ratio = len(ret_hmi) / (window_size * 1.0)
    if hmi_fill_ratio < 0.8 and not args.realtime:
        print 'too few HMI data'
        return
    epoch+=1
    nowmsg=datetime.datetime.strftime(now, '%Y-%m-%d %H:%M:%S')
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

    # clear the goes memoization table.
    global goes_range_max_inner_memo
    goes_range_max_inner_memo = dict()

    while False: # Test code for goes_range_max
        b = random.randrange(window_size)
        e = random.randrange(window_size)
        if not b<e : continue
        print goes_range_max(b,e), max(target_data[b:e])


    # start BPTT learning anew
    state = make_initial_state()

    accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
    if args.backprop_length == 'accel':
        n_backprop = int(2**min(10,int(1+0.05*epoch)))
    else:
        n_backprop = int(args.backprop_length)
    print 'backprop length = ', n_backprop

    last_t = window_size - 24*t_per_hour - 1
    noise_switch = False
    for t in range(last_t+1): # future max prediction training
        sys.stdout.flush()
        time_current = time_begin + t * dt
        learning_stop_time =  last_t - 24*t_per_hour

        input_batch = np.array([feature_data[t]], dtype=np.float32)

        # erase the data
        if t >= learning_stop_time or noise_switch:
            input_batch *= 0.0

        # train to ignore the erased data
        if t % 400==300:
            noise_switch = True
        if t % 400 == 399:
            noise_switch = False
        if t >= learning_stop_time -300:
            noise_switch = False

        output_data = []
        for i in range(24):
            output_data.append(goes_range_max(t+t_per_hour*i,t+t_per_hour*(i+1)))
        for i in range(24):
            output_data.append(goes_range_max(t,t+t_per_hour*(i+1)))
        output_batch = np.array([output_data], dtype=np.float32)

        # count the population in order to learn from imbalanced number of events.
        for i in range(n_outputs):
            poptable[i].add_event(output_data[i])

        input_variable=chainer.Variable(to_PU(input_batch))
        output_variable=chainer.Variable(to_PU(output_batch))

        state, output_prediction = forward_one_step(input_variable, state, train = not args.realtime)

        # accumulate the gradient, modified by the factor
        output_prediction_data = from_PU(output_prediction.data)
        fac = []
        for i in range(n_outputs):
            b,a = poptable[i].population_ratio(output_prediction_data[0, i])
            is_overshoot = output_prediction_data[0, i] >= output_data[i]
            if output_data[i] == encode_goes(0) or output_data[i] == None:
                factor=0.0
            else:
                factor = 1.0/b if is_overshoot else 1.0/a
            # Other options to be considered.
            if args.grad_factor == 'plain':
                factor = 1.0
            if args.grad_factor == 'severe':
                factor = 10.0 ** max(0.0,min(2.0,output_data[i]-4))
            fac.append(factor)

        fac_variable = to_PU(np.array([fac], dtype=np.float32))
        loss_iter = F.sum(fac_variable * abs(output_variable - output_prediction))/float(len(fac))

        # Teach the order of future max prediction
        _, prediction_smaller, _ = F.split_axis(output_prediction, [24,47], axis=1)
        _, prediction_larger     = F.split_axis(output_prediction, [25], axis=1)
        loss_iter_2 = F.sum(F.relu(prediction_smaller - prediction_larger))

        accum_loss += loss_iter ## + 1e-4 * loss_iter_2

        # collect prediction statistics
        if not args.realtime and t >= learning_stop_time:
            for i in range(n_outputs):
                for c in flare_classes:
                    thre = flare_threshold[c]
                    p = output_prediction_data[0, i] >= thre
                    o = output_data[i] >= thre
                    contingency_tables[i,c].add(p,o)
                if i==n_outputs-1:
                    prediction_trace.append([time_current, decode_goes(output_prediction_data[0, i]), decode_goes(output_data[i])])

        # learn
        if t >= learning_stop_time:
            accum_loss.unchain_backward()
        elif (t+1) % n_backprop == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()
            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (t&(t-1)==0 or t%1024==0) and t>0 and t%64==0:
            print 't=',t,' loss=', loss_iter.data, loss_iter_2.data
        if t==last_t-1:
            for j in [0,4,23]:
                i = j+24
                pred_len = j+1
                print '{}hr:'.format(pred_len),
                for c in flare_classes:
                    print '{} {}'.format(c,contingency_tables[i,c].tss()),
            print
            print ppr_contingency_table(contingency_tables)

        if args.realtime == 'quick': break

    if not args.realtime: # at the end of the loop
        print 'dumping...',
        with open('model.pickle','w') as fp:
            if args.gpu >= GPU_STRIDE:
                print 'deepcopy...',
                model_cpu = copy.deepcopy(model).to_cpu()
            else:
                model_cpu = model
            pickle.dump(model_cpu,fp,protocol=-1)
        with open('poptable.pickle','w') as fp:
            pickle.dump(poptable,fp,protocol=-1)
        with open('contingency_tables.pickle','w') as fp:
            pickle.dump(contingency_tables,fp,protocol=-1)
        with open('prediction_trace.pickle','w') as fp:
            pickle.dump(prediction_trace,fp,protocol=-1)
        print 'dump done'

    if args.realtime:
        # visualize forecast
        forecast = Forecast()
        forecast.goes_curve_t = [time_begin + i*dt for i in range(window_size)]
        forecast.goes_curve_y = [decode_goes(target_data[i]) if target_data[i] != encode_goes(0) else None for i in range(window_size)]

        pred_data = output_prediction_data[0]
        pred_curve_t = []
        pred_curve_y = []
        for i in range(24):
            pred_begin_t = now + t_per_hour*i*dt
            pred_end_t   = now + t_per_hour*(i+1)*dt
            pred_flux = decode_goes(pred_data[i])
            pred_curve_t += [pred_begin_t, pred_end_t]
            pred_curve_y += [pred_flux,pred_flux]
        forecast.pred_curve_t = pred_curve_t
        forecast.pred_curve_y = pred_curve_y
        forecast.pred_max_t = []
        forecast.pred_max_y = []
        for i in range(24):
            pred_begin_t = now
            pred_end_t   = now + t_per_hour*(i+1)*dt
            pred_flux = decode_goes(pred_data[i+24])
            max_line_t = [pred_begin_t, pred_end_t]
            max_line_y = [pred_flux,pred_flux]
            forecast.pred_max_t.append(max_line_t)
            forecast.pred_max_y.append(max_line_y)

        archive_dir = now.strftime('archive/%Y/%m/%d')
        subprocess.call('mkdir -p ' + archive_dir, shell=True)
        archive_fn  = archive_dir+now.strftime('/%H%M%S.pickle')

        # pickle, then read from the file, to best ensure the reproducibility.
        print "archiving to: ", archive_fn
        if args.realtime != 'quick':
            with open(archive_fn,'w') as fp:
                pickle.dump(forecast,fp,protocol=-1)
            with open(archive_fn,'r') as fp:
                forecast = pickle.load(fp)


        pngfn = 'prediction-result.png'
        forecast.generate_ccmc_submission()
        forecast.visualize(pngfn)
        subprocess.call('cp {} ~/public_html/'.format(pngfn), shell=True)


        exit(0)


if args.learn_interval == 'random':
    while True:
        learn_predict_from_time(6*365*24 * random.random())
        sys.stdout.flush()
else:
    delta_hour = 24*30
    hour_interval = int(args.learn_interval)
    while delta_hour < 4 * 365 * 24:
        learn_predict_from_time(delta_hour)
        delta_hour += hour_interval
        sys.stdout.flush()
