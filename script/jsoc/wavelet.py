## Wavelet handling module.

import pywt
import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base


def db_fill_columns(self,t_tai, img):
    self.t_tai = t_tai
    for s in ['S','NS']:
        wv_img = wavedec2_img(img, self.wavelet_name, s)
        for r in subspaces(s):
            subimg = wv_img[r.t : r.b , r.l : r.r]
            key = "mean_" + s + "_" + r.key_string()
            val = np.sum(subimg)/r.area()
            setattr(self, key, val)
            key = "mean_sq_" + s + "_" + r.key_string()
            val = np.sum(subimg**2)/r.area()
            setattr(self, key, val)

def db_class_name(series_name, wavelet):
    return "DB_" +(series_name + "_wavelet_" + wavelet).replace('.','_')

def db_class(series_name, wavelet):
    table_name = db_class_name(series_name, wavelet)
    Base = declarative_base()
    class Ret(Base):
        __tablename__ = table_name
        t_tai = sql.Column(sql.DateTime, primary_key=True)
        wavelet_name = wavelet

    for s in ['S','NS']:
        for r in subspaces(s):
            key = "mean_" + s + "_" + r.key_string()
            setattr(Ret, key, sql.Column(sql.Float))
            key = "mean_sq_" + s + "_" + r.key_string()
            setattr(Ret, key, sql.Column(sql.Float))
    Ret.fill_columns = db_fill_columns
    return Ret





class WaveletSubspace:
    def __init__ (self,l,t,r,b):
        # left, top, right, bottom
        self.l=l; self.t=t; self.r=r; self.b=b
    def key_string(self):
        return '{:04}_{:04}_{:04}_{:04}'.format(self.l,self.t,self.r,self.b)
    def area(self):
        return (self.r-self.l)*(self.b-self.t)
    def overwrap(self, other):
        if self.r <= other.l : return False
        if self.l >= other.r : return False
        if self.b <= other.t : return False
        if self.t >= other.b : return False
        return True

def subspaces(w2d_ordering):
    ret = []
    if w2d_ordering=='S':
        for iy in range(11):
            for ix in range(11):
                l=int(2**(ix-1)); t=int(2**(iy-1));
                r=2**ix; b=2**iy;
                ret.append(WaveletSubspace(l,t,r,b))
    elif w2d_ordering=='NS':
        ret.append(WaveletSubspace(0,0,1,1))
        for i in range(10):
            lo=int(2**i)
            hi=int(2**(i+1))
            ret.append(WaveletSubspace(lo,0 ,hi,lo))
            ret.append(WaveletSubspace(0 ,lo,lo,hi))
            ret.append(WaveletSubspace(lo,lo,hi,hi))
    else:
        raise Exception('w2d_ordering should be either "S" or "NS"; got : ' + w2d_ordering)
    return ret


if __name__ == '__main__':
    # test if subspaces don't overwrap, and covers all area
    overwrap_found=False
    for s in ['S','NS']:
        sum = 0
        ss = subspaces(s)
        for r in ss:
            sum += r.area()
        assert( sum == 1024**2)

        for i in range(len(ss)):
            for j in range(i+1,len(ss)):
                if ss[i].overwrap(ss[j]) : overwrap_found=True
    assert(not overwrap_found)

def concat_w2d(ws):
    if (len(ws)==1) : return ws[0]

    ca = ws[0]
    ch,cv,cd = ws[1]

    cah = np.concatenate((ca,ch), axis=0)
    cvd = np.concatenate((cv,cd), axis=0)

    ca2 = np.concatenate((cah,cvd), axis=1)

    return concat_w2d([ca2] + ws[2:])

def wavedec2_img(img, wavelet, w2d_ordering):
    if w2d_ordering=='NS':
        ws = pywt.wavedec2(img,'haar')
        return concat_w2d(ws)
    elif w2d_ordering == 'S':
        for t in range(2):
            for y in range(np.shape(img)[1]):
                img[y] = np.concatenate(pywt.wavedec(img[y], wavelet))
            img = np.transpose(img)
        return img
    else:
        raise Exception('w2d_ordering should be either "S" or "NS"; got : ' + w2d_ordering)
