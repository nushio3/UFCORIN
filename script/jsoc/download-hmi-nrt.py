#!/usr/bin/env python

import glob, os, pywt, re, shutil, subprocess, sys
from astropy.io import fits
import numpy as np
import scipy.ndimage.interpolation as intp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def system(cmd):
    subprocess.call(cmd, shell=True)


# seems to downlowd the SDO/HMI fits file for given years.

wl = '/mnt'
yearstart = 2015
monthstart = 7
yearend = 2015
monthend = 8
bucket = "sdo"

path= '/home/ubuntu/hub/UFCORIN/script/jsoc/'

workdir="nrt"
if not os.path.exists(workdir): os.mkdir(workdir)
os.chdir(workdir)

query = "hmi.M_720s_nrt[$]"
command = path+"exportfile.csh '"+query+ "' " + sys.argv[1]
print command
#system(command)

def fits2npz(newfn, npzfn):
    hdulist=fits.open(newfn)
    img=hdulist[1].data
    img = np.where( np.isnan(img), 0.0, img)
    img2=intp.zoom(img,zoom=zoom_ratio)

    for y in range(reso_new):
        for x in range(reso_new):
            x0=reso_new/2.0-0.5
            y0=reso_new/2.0-0.5
            r2 = (x-x0)**2 + (y-y0)**2
            r0 = 1800.0*zoom_ratio
            if r2 >= r0**2 : img2[y][x]=0.0

    np.savez_compressed(npzfn, img=np.float32(img2))
    register_wavelet(img2, newfn.replace('.fits','.png'))
        
def concat_w2d(ws):
    if (len(ws)==1) : return ws[0]

    ca = ws[0]
    ch,cv,cd = ws[1]

    cah = np.concatenate((ca,ch), axis=0)
    cvd = np.concatenate((cv,cd), axis=0)

    ca2 = np.concatenate((cah,cvd), axis=1)
    
    return concat_w2d([ca2] + ws[2:])

def plot_img(img,fn,title_str):
    w,h= np.shape(img)
    dpi=200
    plt.figure(figsize=(8,6),dpi=dpi)
    fig, ax = plt.subplots()
	
    cmap = plt.get_cmap('bwr')
    cax = ax.imshow(img,cmap=cmap,vmin=-100.0,vmax=100.0)  # extent=(0,w,0,h),
    cbar=fig.colorbar(cax)
    ax.set_title(title_str)
    fig.savefig(fn,dpi=dpi)
    plt.close('all')

def wavedec2std(img, wavelet):
    for t in range(2):
        for y in range(np.shape(img)[1]):
            img[y] = np.concatenate(pywt.wavedec(img[y], wavelet))
        img = np.transpose(img)
    return img

def register_wavelet(img, imgfn):
    print img.shape
    ws = pywt.wavedec2(img,'haar')
    wavelet_img = concat_w2d(ws)
    plot_img(img,imgfn,"real")
    plot_img(wavelet_img,"NS_" + imgfn,"NS-wavelet")
    wavelet_img = wavedec2std(img,'haar')
    plot_img(wavelet_img,"S_" + imgfn,"S-wavelet")
    

for fn in glob.glob('*.fits'):
    if not re.match('hmi',fn): continue

    ma = re.search('%5B(\d+)\.(\d+)\.(\d+)_(\d+)%3A(\d+)',fn)
    if not ma: continue
    yyyy=ma.group(1)
    mm=ma.group(2)
    dd=ma.group(3)
    hh=ma.group(4)
    minu=ma.group(5)
    
    print 'detect {}-{}-{} {}:{}'.format(yyyy,mm,dd,hh,minu)

    newfn=hh+minu+'.fits'
    shutil.copy(fn,newfn)
    s3 = "aws s3 cp "+newfn+" s3://sdo/hmi/mag720/"+yyyy+"/"+mm+"/"+dd+"/"+newfn
    print s3
    system(s3)

    reso_original=4096
    reso_new=1024
    zoom_ratio = float(reso_new)/reso_original

    npzfn = newfn.replace('.fits','.npz')
    fits2npz(newfn, npzfn)

    cmd='aws s3 cp '+npzfn+ (' s3://sdo/hmi/mag720x{}/{}/{}/{}/'.format(reso_new,yyyy,mm,dd))+npzfn
    print cmd
    system(cmd)



