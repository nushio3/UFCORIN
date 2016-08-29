#!/usr/bin/env python
# -*- coding: utf-8 -*-

import astropy.time as time
import calendar, datetime, os, requests, subprocess,urllib, sys, StringIO
import numpy as np
from astropy.io import fits
import sunpy

global data_path
data_path = os.environ['ufcorin_bigdata_path']


# 時刻tにおける太陽磁場画像を取得します
# これは1024^2に縮小されたデータです。
# SDO衛星が撮影した元データは http://sdo.gsfc.nasa.gov/data/ にあります。
def get_hmi_image(t):
    global data_path
    if data_path is not None:
        fn = data_path + t.strftime('/hmi/%Y/%m/%d/%H%M.npz').replace('/',os.sep)
        try:
            return np.load(fn)['img']
        except:
            return None


    try:
        url2 = 'http://jsoc2.stanford.edu/data/hmi/fits/{:04}/{:02}/{:02}/hmi.M_720s_nrt.{:04}{:02}{:02}_{:02}{:02}00_TAI.fits'.format(t.year, t.month, t.day, t.year, t.month, t.day, t.hour, t.minute)
        url = 'http://jsoc2.stanford.edu/data/hmi/fits/{:04}/{:02}/{:02}/hmi.M_720s.{:04}{:02}{:02}_{:02}{:02}00_TAI.fits'.format(t.year, t.month, t.day, t.year, t.month, t.day, t.hour, t.minute)

        resp = requests.get(url)
        if resp.status_code != 200:
            resp = requests.get(url2)
        strio = StringIO.StringIO(resp.content)

        hdulist=fits.open(strio)
        hdulist.verify('fix')
        img=hdulist[1].data
        img = np.where( np.isnan(img), 0.0, img)

        return img
    except Exception as e:
        sys.stderr.write(str(e.message))
        return None


# 時刻tにおける、波長wavelengthの太陽画像を取得します
# これは1024^2に縮小されたデータです。
# SDO衛星が撮影した元データは http://sdo.gsfc.nasa.gov/data/ にあります。
def get_aia_image(wavelength,t):
    global data_path
    cache_fn = data_path + t.strftime('/aia/%Y/%m/%d/%H%M.fits').replace('/',os.sep)
    if os.path.exists(cache_fn):
        try:
            return sunpy.map.Map(cache_fn)
        except:
            return None


    url = 'http://jsoc2.stanford.edu/data/aia/synoptic/{:04}/{:02}/{:02}/H{:02}00/AIA{:04}{:02}{:02}_{:02}{:02}_{:04}.fits'.format(t.year, t.month, t.day,t.hour, t.year, t.month, t.day, t.hour, t.minute, wavelength)
    url2 = 'http://jsoc2.stanford.edu/data/aia/synoptic/nrt/{:04}/{:02}/{:02}/H{:02}00/AIA{:04}{:02}{:02}_{:02}{:02}00_{:04}.fits'.format(t.year, t.month, t.day,t.hour, t.year, t.month, t.day, t.hour, t.minute, wavelength)
        
    resp = requests.get(url)
    if resp.status_code != 200:
        resp = requests.get(url2)
        if resp.status_code != 200:
            stderr.write("no file found: {} {}".format(url, url2))
            return None
    strio = StringIO.StringIO(resp.content)
    subprocess.call(['mkdir','-p',os.path.dirname(cache_fn)])
    with open(cache_fn,'w') as fp:
        fp.write(strio.read())
    return sunpy.map.Map(cache_fn)



global goes_raw_data_fast
goes_raw_data_fast = None
# 時刻t0におけるgoes X線フラックスの値を返します。
# １時間ごとの精度しかありませんが、高速です
def get_goes_flux_fast(t0):
    global goes_raw_data_fast
    if goes_raw_data_fast is None:
        goes_raw_data_fast = {}
        with open("goes-data-1hr.txt","r") as fp:
            for l in iter(fp.readline, ''):
                ws = l.split()
                t = datetime.datetime.strptime(ws[0],"%Y-%m-%dT%H:%M")
                x = float(ws[1])
                goes_raw_data_fast[t] = x
    t=datetime.datetime(t0.year,t0.month,t0.day,t0.hour)
    if t in goes_raw_data_fast:
        return goes_raw_data_fast[t]
    return 1e-8

# 時刻t0におけるgoes X線フラックスの値を返します。
# １時間ごとの精度しかありませんが、高速です
def get_goes_max_fast(t, timedelta):
    ret = get_goes_flux_fast(t)
    dt = datetime.timedelta(0)
    while dt <= timedelta:
        ret = max(ret, get_goes_flux_fast(t + dt))
        dt += datetime.timedelta(hours = 1)
    return ret



global goes_raw_data, goes_loaded_files
goes_raw_data = {}
goes_loaded_files = set()

# 時刻t0におけるgoes X線フラックスの値を返します。
def get_goes_flux(t0):
    global goes_raw_data, goes_loaded_files, data_path
    if t0 in goes_raw_data:
        return goes_raw_data[t0]

    day31 = calendar.monthrange(t0.year,t0.month)[1]
    fn = 'g15_xrs_1m_{y:4}{m:02}{d:02}_{y:4}{m:02}{d31:02}.csv'.format(y=t0.year, m=t0.month, d=01, d31=day31)
    localpath = os.path.join(data_path , fn)
    if localpath in goes_loaded_files:
        return None

    if not(os.path.exists(localpath)):
        url = 'http://satdat.ngdc.noaa.gov/sem/goes/data/new_avg/{y}/{m:02}/goes15/csv/'.format(y=t0.year, m=t0.month) + fn
        resp = urllib.urlopen(url)
        with open(localpath,'w') as fp:
            fp.write(resp.read())
    if not(os.path.exists(localpath)):
        return None

    goes_loaded_files.add(localpath)
    with (open(localpath, "r")) as fp:
        while True:
            con = fp.readline()
            if con[0:5]=='data:':
                break
        fp.readline()

        while True:
            con = fp.readline()
            if con=='':
                break
            ws = con.split(',')
            t = time.Time(ws[0]).datetime
            goes_raw_data[t] = float(ws[6])

    return get_goes_flux(t)

goes_max_epoch = time.Time("2000-01-01 00:00:00").datetime

# t から　timedeltaの時間のあいだのgoes X線フラックスの最大値を返します。
def get_goes_max(t, timedelta):
    i = int((t - goes_max_epoch).total_seconds())
    j = int((t - goes_max_epoch + timedelta).total_seconds())
    start=min(i,j)
    end  =max(i,j)
    delta=1;
    while delta < end:
        delta*=2
    return goes_max_for_secondrange(delta,start, end)

global goes_max_for_secondrange_memo
goes_max_for_secondrange_memo={}
def goes_max_for_secondrange(delta, start ,end):
    global goes_max_for_secondrange_memo

    box0 = delta*(int(start + delta-1)/int(delta))
    box1 = delta*(int(end)/int(delta))
    key = (start,end)


    if start >= end:
        return None
    if end-start < 60:
        t = goes_max_epoch + datetime.timedelta(seconds = (int(start)/60)*60)
        return get_goes_flux(t)
    if key in goes_max_for_secondrange_memo:
        return goes_max_for_secondrange_memo[key]

    if box0 > box1:
        return goes_max_for_secondrange(delta/2, start, end)
    if box0 == box1:
        return max(goes_max_for_secondrange(delta/2, start, box0),
                   goes_max_for_secondrange(delta/2, box0, end))
    bm_mid = max(goes_max_for_secondrange(delta/2, box0, box0+delta/2)
                 ,goes_max_for_secondrange(delta/2, box0+delta/2, box1))
    goes_max_for_secondrange_memo[(box0, box1)]=bm_mid

    return  max(goes_max_for_secondrange(delta/2, start, box0),
                bm_mid,
                goes_max_for_secondrange(delta/2, box1, end))
