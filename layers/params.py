import mxnet as mx
from mxnet import nd
from mxboard import SummaryWriter

gamma = 0.0000125
power = 1
c_rate = -0.9
iter_stop = 450000

root = 'log_bn_dy2'
sw = SummaryWriter(logdir=root, flush_secs=5)
kept_in_kernel = 3

alpha = 1.0  # second punishment coeffcient


class mask_param(object):
    def __init__(self, iter=0):
        super(mask_param, self).__init__()
        self.netMask = {}
        self.iter = iter

    def set_param(self, keys, ctx=mx.cpu()):
        self.netMask = dict(zip(keys, nd.array([1] * len(keys), ctx=ctx)))

    def load_param(self, mp, ctx=mx.cpu()):
        self.iter = mp.iter
        for k, v in mp.netMask.items():
            self.netMask[k] = v.as_in_context(ctx)


# this class was prepare the paramers in carry out activiation
class dropout_paramer(object):
    def __init__(self):
        super(dropout_paramer, self).__init__()
        self.selected = {}

    def init_select(self, mask, key, ratio):
        length = mask.shape[0]
        select = int(length * ratio)
        idx = nd.random.shuffle(nd.arange(length))[:select].astype('int32').asnumpy()
        Dmask = nd.zeros_like(mask).asnumpy()
        Dmask[idx] = 1
        Dmask = nd.array(Dmask).as_in_context(mask.context)
        self.selected[key] = Dmask

    def get_select(self, mask, key, ratio):
        if global_param.iter == 0:
            self.init_select(mask, key, ratio)
        return self.selected[key]


global_dropout = dropout_paramer()
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
