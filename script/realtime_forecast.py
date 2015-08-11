#!/usr/bin/env python

import datetime, math, os, random
import astropy.time as time
import numpy as np
import sqlalchemy as sql
from   sqlalchemy.orm import sessionmaker
from   sqlalchemy.ext.declarative import declarative_base

import goes.schema as goes
import jsoc.wavelet as wavelet

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

print n_feature


goes_range_max_inner_memo = dict()

def encode_goes(x):
    return 10 + math.log(max(1e-10,x))/math.log(10.0)
def decode_goes(x):
    return 10.0**(x-10)

def encode_hmi(x):
    return math.log(max(1.0,x))
def decode_goes(x):
    return math.exp(x)



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
