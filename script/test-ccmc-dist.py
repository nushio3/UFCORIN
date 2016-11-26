#!/usr/bin/env python

import random,math


mu = 0
vv = 0
w=0.2
for t in range(10100):
    x = random.random() * 100
    if t == 10000:
        print "boom"
    if t > 10000:
        x = 500
    mu = (1-w) * mu + w * x
    vv = (1-w) * vv + w * x**2
    sigma = math.sqrt(vv - mu**2)
    print mu, sigma


# standard deviation of uniform distribution
# > 100/ sqrt 12
# 28.86751345948129
