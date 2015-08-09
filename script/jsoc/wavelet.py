## Wavelet handling module.

import pywt
import numpy as np

def concat_w2d(ws):
    if (len(ws)==1) : return ws[0]

    ca = ws[0]
    ch,cv,cd = ws[1]

    cah = np.concatenate((ca,ch), axis=0)
    cvd = np.concatenate((cv,cd), axis=0)

    ca2 = np.concatenate((cah,cvd), axis=1)
    
    return concat_w2d([ca2] + ws[2:])

def wavedec2_img_NS(img, wavelet):
    ws = pywt.wavedec2(img,'haar')
    return concat_w2d(ws)

def wavedec2_img_S(img, wavelet):
    for t in range(2):
        for y in range(np.shape(img)[1]):
            img[y] = np.concatenate(pywt.wavedec(img[y], wavelet))
        img = np.transpose(img)
    return img

