#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import sys
import contingency_table

with open(sys.argv[1],'r') as fp:
    ct = pickle.load(fp)
tbl=ct[47,'>=M']
for p in [False,True]:
    for o in [False,True]:
        print 'p={} o={} : {}'.format(p,o, tbl.counter[p,o])
print tbl.tss()
