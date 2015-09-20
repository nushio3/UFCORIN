#!/usr/bin/env python

# print 'export OPENBLAS_NUM_THREADS=1'
for r in [1., 3., 0.3, 10., 0.1]:
    for bplen in ['accel','2','32', '1024']:
        for gf in ['ab','flat','severe']:
            opts = '--backprop-length {} --grad-factor {}'.format(bplen, gf)
            print "-o AdaDelta -p '(rho={})' {}".format(1.0 - 0.05*r, opts)
            print "-o AdaGrad -p '(lr={})' {}".format(0.001*r, opts)
            print "-o Adam -p '(alpha={})' {}".format(0.001*r, opts)
            print "-o MomentumSGD -p '(lr={})' {}".format(0.01*r, opts)
            print "-o RMSprop -p '(lr={})' {}".format(0.01*r, opts)
            print "-o SGD -p '(lr={})' {}".format(0.01*r, opts)
 
