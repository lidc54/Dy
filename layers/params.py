import mxnet as mx
from mxnet import nd
from mxboard import SummaryWriter
import math

gamma = 0.0000025  # about 10w to 1
power = 1
c_rate = -0.9
iter_stop = 4500000
nums_power = 1
advanced_iter = 0.5  # defalut is 1:not to advanced
zoom = 5.0

root = 'log_4dy_Ns2'
sw = SummaryWriter(logdir=root, flush_secs=5)
kept_in_kernel = 3

alpha = 1  # harder punishment for loss


class mask_param(object):
    def __init__(self, iter=0):
        super(mask_param, self).__init__()
        self.netMask = {}
        self.iter = iter
        self.Is_kept_ratio = 0.9  # prefer the nums in kernel equal to kept_int_kernel
        self.kept_ratio = 0.0

    def set_param(self, keys, ctx=mx.cpu()):
        self.netMask = dict(zip(keys, nd.array([1] * len(keys), ctx=ctx)))

    def get_kept_ratio(self):
        self.kept_ratio = self.Is_kept_ratio * (1 - math.pow(1 + gamma * self.iter, -power))
        return self.kept_ratio

    def load_param(self, mp, ctx=mx.cpu()):
        self.iter = mp.iter
        for k, v in mp.netMask.items():
            self.netMask[k] = v.as_in_context(ctx)


# mask and iter times for convlution of BatchNorm
# so mask type has a scope of ['conv','bn']
global_param = mask_param()


def get_params():
    import pickle
    loaded_param = "../model_paramers/global.param"
    with open(loaded_param)as f:
        sv = pickle.load(f)

    print('o')


if __name__ == "__main__":
    get_params()
