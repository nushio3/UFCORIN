#!/usr/bin/env python

for r in [1., 3., 0.3, 10., 0.1]:

    print "-o AdaDelta -p '(rho={})'".format(1.0 - 0.05*r)
    print "-o AdaGrad -p '(lr={})'".format(0.001*r)
    print "-o Adam -p '(alpha={})'".format(0.001*r)
    print "-o MomentumSGD -p '(lr={})'".format(0.01*r)
    print "-o RMSprop -p '(lr={})'".format(0.01*r)
    print "-o SGD -p '(lr={})'".format(0.01*r)
