#!/usr/bin/env python

import sys,os, glob, shutil

# seems to downlowd the SDO/HMI fits file for given years.

wl = 'mag'
yearstart = 2015
monthstart = 6
yearend = 2015
monthend = 7
bucket = "sdo"

path= '/home/ubuntu/hub/UFCORIN/script/jsoc/'

if not os.path.exists(wl): os.mkdir(wl)
os.chdir(wl)
for i in range(yearstart,yearend+1):
    year='%04d' % i
    if not os.path.exists(year): os.mkdir(year)
    os.chdir(year)
    #s3dir = "/user/shibayama/sdo/aia/"+wl+"/"+year
    #os.system(s3)

    for j in range(1,13):
        if (i==yearstart and j<monthstart):
            continue
        if (i==yearend and j==monthend):
            break

        month='%02d' % j 
        if not os.path.exists(month): os.mkdir(month)
        os.chdir(month)
        if j!=12:
            query = "hmi.M_45s["+year+'.'+month+".1-"+year+'.'+month+".15@1h]"
            command = path+"exportfile.csh "+query+ " muranushi@gmail.com"
            print command
            os.system(command)
            query = "hmi.M_45s["+year+'.'+month+".15-"+year+'.'+str(j+1)+"@1h]"
        else :
            query = "hmi.M_45s["+year+'.'+month+".1-"+year+'.'+month+".15@1h]"
            command = path+"exportfile.csh "+query+ " muranushi@gmail.com"
            print command
            os.system(command)
            query = "hmi.M_45s["+year+'.'+month+".15-"+str(i+1)+".01@1h]"
        command = path+"exportfile.csh "+query+ " muranushi@gmail.com"
        print command
        os.system(command)

        for fn in glob.glob('*.fits'):
            yyyy=fn[12:16]
            mm=fn[17:19]
            dd=fn[20:22]
            hh=fn[23:25]

            print yyyy,mm,dd,hh

            if mm!=month: 
                os.remove(fn)
                continue
            if not os.path.exists(dd): os.mkdir(dd)
            name=dd+'/'+hh+'.fits'
            shutil.move(fn,name)
        os.chdir('..')
        s3 = "aws s3 sync "+month+" s3://sdo/hmi/"+wl+"/"+year+"/"+month+"/"
        print s3
        os.system(s3)
        shutil.rmtree(month)
    os.chdir('..')

