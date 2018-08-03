'''
this file is to train a model to learn location with a different way
'''

import mxnet.gluon.nn as nn
import numpy as np
from mxnet import nd
import mxnet as mx
from layers.params import global_param


# import mxnet.gluon as gluon
# # import gradcam, pickle, dataset
# import math
# from layers import data
# from layers.dy_conv import origin_conv


class new_location_conv(nn.Conv2D):
    def __init__(self, channels, kernel_size, **kwargs):
        super(new_location_conv, self).__init__(channels, kernel_size, **kwargs)
        self.reidx = (np.arange(9) + 1) % 9

    def hybrid_forward(self, F, x, weight, bias=None):
        Cout, Cin, k1, k2 = weight.shape
        wmask = nd.topk(nd.abs(weight).reshape(Cout, Cin, -1), k=3, ret_typ='mask')
        wmask = (wmask == 0) * 0.1 + wmask
        wmask = wmask.reshape(Cout, Cin, k1, k2).as_in_context(x.context)

        # ratio = 1 - 0.5 * global_param.get_kept_ratio()
        # wmask = np.ones((Cout, Cin, k1 * k2))
        # wmask[:, :, self.mask] = ratio
        # name = self.name + '_weight'
        # global_param.netMask[name] = wmask

        temp_weight = weight.reshape((Cout, Cin, -1))
        new_weight = temp_weight[:, :, self.reidx].reshape(Cout, Cin, k1, k2)
        new_weight = (weight + nd.sign(weight) * nd.abs(new_weight))

        # ratio=0.2 if ratio<0.2 else ratio
        return super(new_location_conv, self).hybrid_forward(F, x, new_weight * wmask, bias)


if __name__ == "__main__":
    a = new_location_conv(64, 3, in_channels=3)
    from compress_model.general_conv import init_s

    ctx = mx.gpu()
    init_s(a, ctx=ctx)
    b = nd.random.uniform(-1, 1, shape=(1, 3, 5, 5)).as_in_context(ctx)
    a(b)
