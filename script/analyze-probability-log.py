#!/usr/bin/env python2

observation_map = {}
probability_map = {}

ts = set()

with open('archive/observation_log.txt','r') as fp:
    con = fp.read()
    for l in con.split("\n"):
        for w in l.split():
            if len(w) < 2:
                continue
                t = int(w[0])
                x = int(w[1])
                ts.add(t)
                observation_map[t] = x


with open('archive/probability_log.txt','r') as fp:
    con = fp.read()
    for l in con.split("\n"):
        for w in l.split():
            if len(w) < 4:
                continue
                t = int(w[0])
                ts.add(t)
                px = int(w[1])
                pm = int(w[2])
                pc = int(w[3])

                probability_map[t] = {}
                probability_map[t][1e-4] = px
                probability_map[t][1e-5] = pm
                probability_map[t][1e-6] = pc

for flare_class in [1e-4, 1e-5, 1e-6]:
    histogram_den = 10 * [1e-20]
    histogram_num = 10 * [0]
    for t in ts:
        if t not in observation_map or t not in probability_map:
            continue
        goes_flux = observation_map[t]
        p = probability_map[t][flare_class]
        assert(0<=p and p <=1)
        p_class = int(p*10)
        histogram_den[p_class] += 1
        if goes_flux >= flare_class:
            histogram_num[p_class] += 1
