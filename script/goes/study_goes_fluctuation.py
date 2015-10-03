#!/usr/bin/env python
import urllib2
import math

def classify(x):
    return floor(math.log(x) / math.log(10))


caseJump = 0
caseFlat = 0

def judge(vals):
    c0 = classify(vals[0])
    for v in vals[1:]:
        if classify(v) != c0:
            caseJump += 1
            return
    caseFlat += 1
    return

def analyze(con):
    data_part=False
    vals = []
    for l in con.split('\n'):
        if l.strip() == 'data:' :
            data_part=True
        if data_part:
            words = l.split(',')
            if len(words) < 7 : continue
            match = re.search('^(\d+)-(\d+)-(\d+)\s+(\d+):(\d+)',words[0])
            if not match :
                continue
            a_qf = int(words[1])
            b_qf = int(words[4])
            if a_qf != 0 or b_qf !=0 : continue

            flux_long = float(words[6])
            vals.append(flux_long)

            if len(vals) >= 30:
                judge(vals)
                vals=[]

for year in range(2011,2012):
    for month in range(1,13):
        for day in range(1,32):
            url='http://satdat.ngdc.noaa.gov/sem/goes/data/new_full/{y}/{m:02d}/goes15/csv/g15_xrs_2s_{y}{m:02d}{d:02d}_{y}{m:02d}{d:02d}.csv'.format(y=year,m=month,d=day)
            try:
                fp = urllib2.urlopen(url)
                analyze(fp.read())
                fp.close()
            except:
                pass
            print year,month,day, caseJump, caseFlat
