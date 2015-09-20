#!/usr/bin/env python

import argparse, glob, os, re

for path in  glob.glob('result/*'):
    log_fn = path + '/stdout.txt'
    args_fn = path + '/args.log'
    if not os.path.exists(log_fn): continue
    if not os.path.exists(args_fn): continue

    with open(args_fn) as fp:
        result = eval('argparse.' + fp.read())

    with open(log_fn) as fp:
        log = fp.read()
        if re.search('epoch\=400', log):
            progress = "DONE"
        else:
            progress = "WIP"
            continue
        for l in log.split('\n'):
            ma = re.search('24hr\: (.*)', l)
            if ma:
                last_line = ma.group(1)
        ws = last_line.split()
        
    result.tss_x = float(ws[1])
    result.tss_m = float(ws[3])
    result.tss_c = float(ws[5])

    print result.backprop_length, result.grad_factor, result.optimizer + result.optimizeroptions, result.tss_x, result.tss_m, result.tss_c,path 
