#!/usr/bin/env python

import pywt
import numpy as np

img = np.load('0000.npz')['img']
print img.shape

ws = pywt.wavedec2(img,'haar')

def stat(ar):
    print np.sum(ar), np.sum(ar**2)

stat(ws[0])
for ch,cv,cd in ws[1:] :
    stat(ch)
    stat(cv)
    stat(cd)

s=0
for i in range(1024):
    s+= len(pywt.wavedec(img[i],'haar'))
print s

