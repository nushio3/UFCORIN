#!/usr/bin/env python3

import pickle
import sys

class Forecast:
    pass

for fn in sys.argv:
    with open(fn,"rb") as fp:
        forecast = pickle.load(fp)
    print(dir(forecast))
