
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import numpy as np


xs = chainer.Variable(np.array([i for i in range(100)],dtype=np.float32))
_, smaller, _ = F.split_axis(xs, [24,47], axis=0)
smaller.backward()
