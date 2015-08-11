#!/usr/bin/env python
import goes.schema as goes
import jsoc.wavelet as wavelet

GOES = goes.GOES
HMI  = wavelet.db_class('hmi.M_720s_nrt', 'haar')
