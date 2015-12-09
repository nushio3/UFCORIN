#!/usr/bin/env python

import argparse, glob, os, re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

tss_samples = {'x':{},'m':{},'c':{}}

tss_max_x = 0.0
tss_max_m = 0.0
tss_max_c = 0.0

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
            #continue
        for l in log.split('\n'):
            ma = re.search('24hr\: (.*)', l)
            if ma:
                last_line = ma.group(1)
        ws = last_line.split()

    result.tss_x = float(ws[1])
    result.tss_m = float(ws[3])
    result.tss_c = float(ws[5])

    tss_max_x = max(tss_max_x, result.tss_x)
    tss_max_m = max(tss_max_m, result.tss_m)
    tss_max_c = max(tss_max_c, result.tss_c)

    category = result.grad_factor

    if category not in tss_samples['x']:
        tss_samples['x'][category] = []
        tss_samples['m'][category] = []
        tss_samples['c'][category] = []

    tss_samples['x'][category].append(result.tss_x)
    tss_samples['m'][category].append(result.tss_m)
    tss_samples['c'][category].append(result.tss_c)

    #print progress,result.backprop_length, result.grad_factor, result.optimizer, result.optimizeroptions, result.tss_x, result.tss_m, result.tss_c,path

print tss_max_x,tss_max_m,tss_max_c

exit

f, axarr = plt.subplots(3,3, sharex=True, sharey=True)
mpl.rcParams.update({'font.size': 7})
j = 0
for kc in ['x','m','c']:
    ctr = 0
    for k,v in sorted(tss_samples[kc].iteritems()):
        axarr[ctr][j].hist(v, bins=20, range=(0,1))
        if j==0 : axarr[ctr][j].set_ylabel(k)
        if ctr==5 : axarr[ctr][j].set_xlabel('TSS')
        if ctr==0 : axarr[ctr][j].set_title(kc + ' class')
        axarr[ctr][j].set_ylim((0,20))
        ctr+=1
    j+=1
plt.savefig('results-histogram.png', dpi=200)
plt.close('all')
