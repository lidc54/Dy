"""
this file aims to test an idea:
will the moving filter can get ideal weights
"""

import mxnet as mx
from mxnet import nd, autograd
import numpy as np
import math

x = nd.ones((3, 3))
# x = nd.random.uniform(-1, 1, shape=(3, 3))
# m = nd.zeros((3, 3))
a = (np.arange(9) + 1) % 9
# m[range(3), a] = [2, 3, 1]
x.attach_grad()
y = nd.array([2, 5, 7, 1, 3, 6, 7, 2, 6]).reshape(9, 1)
idx = 0
for i in range(1350):
    with autograd.record():
        m = nd.array(nd.topk(nd.abs(x).reshape(-1), k=3, ret_typ='mask').copy())
        if i == 0:
            m = nd.array(nd.random.shuffle(m).asnumpy())
        new_idx = nd.sum(nd.topk(m, k=3))
        m = m.reshape(3, 3)
        new_x = x.reshape(-1)[a].reshape(3, 3)  # x.T #
        ratio = math.pow((1 + 0.05 * i), -1)
        xx = (x + ratio * new_x)
        z = nd.dot((m * xx).reshape(1, 9), y)
        z = nd.abs(z - 15)
    z.backward()
    x[:] = x - 0.0001 * x.grad
    if idx != new_idx:
        idx = new_idx
        print 'i:', i, idx.asscalar()
        print m.asnumpy().astype(int)
    if i % 50 != 0: continue
    print z.reshape(-1).asscalar(),
print x
