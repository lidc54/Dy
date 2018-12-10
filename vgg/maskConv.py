from mxnet.gluon import nn
from mxnet import nd, initializer, autograd
from mxnet.gluon import loss, Trainer
import mxnet as mx
import math, random

class Maskedconv(nn.Conv2D):
    def __init__(self, channels, kernel_size, **kwargs):
        '''
            the function same as '../layers/dy_conv new_conv
            but the difference is this is more friendly ctx is a list
        '''
        super(Maskedconv, self).__init__(channels, kernel_size, **kwargs)

    def hybrid_forward(self, F, x, weight, bias=None):
        keys = self.params.keys()
        # self.assign_mask(keys)
        key = [key for key in keys if 'weight' in key][0]

        wmask = global_param.netMask[key] = \
            assign_mask(weight, global_param.netMask[key], key)

        bmask = wmask.copy()
        for i in range(2): bmask = nd.sum(bmask, axis=-1)
        # if global_param.iter % 1000 == 0:
        #     tag_key = '_'.join(key.split('_')[1:]) + '_KerLossNums'
        #     gls.sw.add_histogram(tag=tag_key, values=bmask.reshape(-1).copy().sort(), global_step=global_param.iter)
        bmask = nd.sum(bmask, axis=-1) > 0

        return super(Maskedconv, self).hybrid_forward(F, x, weight * wmask, bias * bmask)

