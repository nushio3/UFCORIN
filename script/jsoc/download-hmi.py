#!/usr/bin/env python

import sys,os, glob, shutil, re, subprocess

def system(cmd):
    subprocess.call(cmd, shell=True)


# seems to downlowd the SDO/HMI fits file for given years.

wl = 'mag720'
yearstart = 2011
monthstart = 1
yearend = 2015
monthend = 4
bucket = "sdo"

path= '/home/ubuntu/hub/UFCORIN/script/jsoc/'

if not os.path.exists(wl): os.mkdir(wl)
os.chdir(wl)
for i in reversed(range(yearstart,yearend+1)):
    year='%04d' % i
    if not os.path.exists(year): os.mkdir(year)
    os.chdir(year)
    #s3dir = "/user/shibayama/sdo/aia/"+wl+"/"+year
    #system(s3)

    for j in reversed(range(1,13)):
        if (i==yearstart and j<monthstart):
            continue
        if (i==yearend and j>monthend):
            continue

        month='%02d' % j 
        for k in [1,7,13,19,25]:
            k2=k+6
            if k==25: k2=32
            if not os.path.exists(month): os.mkdir(month)
            os.chdir(month)
            query = "hmi.M_720s[{y}.{m}.{d}-{y}.{m}.{d2}@720s]".format(y=i,m=j,d=k,d2=k2)
            command = path+"exportfile.csh "+query+ " " + sys.argv[1]
            print command
            system(command)
            system('mv jsoc_export.* exportlog-{}-{}-{}.txt'.format(i,j,k))

            for fn in glob.glob('*.fits'):
                if not re.match('hmi',fn): continue
    
                # hmi.M_720s%5B2011.03.05_12%3A00%3A00_TAI%5D%5B1%5D%7Bmagnetogram%7D.fits
                ma = re.search('%5B(\d+)\.(\d+)\.(\d+)_(\d+)%3A(\d+)',fn)
                if not ma: continue
                yyyy=ma.group(1)
                mm=ma.group(2)
                dd=ma.group(3)
                hh=ma.group(4)
                minu=ma.group(5)
    
                print 'detect {}-{}-{} {}:{}'.format(yyyy,mm,dd,hh,minu)
    
                if mm!=month: 
                    os.remove(fn)
                    continue
                if not os.path.exists(dd): os.mkdir(dd)
                name=dd+'/'+hh+minu+'.fits'
                shutil.move(fn,name)
            os.chdir('..')
            s3 = "aws s3 sync "+month+" s3://sdo/hmi/"+wl+"/"+year+"/"+month+"/"
            print s3
            system(s3)
            ## exit(0) # test disk capacity here
            shutil.rmtree(month)

    os.chdir('..')

