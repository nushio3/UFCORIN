#!/usr/bin/env python

import numpy as np


hmi = np.load('aia193/0012.npz')['img']
print np.average(hmi)
print np.sqrt(np.average(hmi**2))
print np.max(hmi)
print np.min(hmi)
