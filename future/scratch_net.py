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
        self.rearray_idx = (np.arange(9) + 1) % 9

    def hybrid_forward(self, F, x, weight, bias=None):
        Cout, Cin, k1, k2 = weight.shape
        temp_weight = nd.abs(weight).reshape((Cout, Cin, -1))
        wmask = nd.array(nd.topk(temp_weight, k=3, ret_typ='mask').copy())
        wmask = wmask.reshape(Cout, Cin, k1, k2)
        name = self.name + '_weight'
        global_param.netMask[name] = wmask
        new_weight = temp_weight[:, :, self.rearray_idx].reshape(Cout, Cin, k1, k2)
        ratio = 1 - 0.5 * global_param.get_kept_ratio()
        new_weight = (weight + ratio * new_weight)
        return super(new_location_conv, self).hybrid_forward(F, x, new_weight * wmask, bias)


if __name__ == "__main__":
    a = new_location_conv(64, 3, in_channels=3)
    from compress_model.general_conv import init_s

    ctx = mx.gpu()
    init_s(a, ctx=ctx)
    b = nd.random.uniform(-1, 1, shape=(1, 3, 5, 5)).as_in_context(ctx)
    a(b)
