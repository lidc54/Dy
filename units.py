from layers.params import global_param, global_dropout
from layers.dy_conv import assign_mask
from mxnet import nd
import mxnet, os


def getParmas(mnet, mode='conv'):
    """
    get paramers of spherenet solely
    :param net: shperenet_20
    :param mode: 'conv','fc','all'
    :return: params of net
    """
    if mode == 'conv':
        params = mnet.net.collect_params()
    elif mode == 'all':
        params = mnet.collect_params()
    elif mode == 'fc':
        params = mnet.f6.collect_params()
    else:
        raise Exception('no such a mode')
    return params


def init_sphere(mnet, loaded_model, ctx=mxnet.cpu()):
    for k, v in mnet.collect_params().items():
        if 'bias' in k:
            v.initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
        elif 'batchnorm' in k:
            if 'gamma' in k or 'var' in k:
                v.initialize(mxnet.initializer.Constant(1.0), ctx=ctx)
            elif 'beta' in k or 'mean' in k:
                v.initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
        else:
            v.initialize(mxnet.initializer.Xavier(magnitude=3), ctx=ctx)
    # load exist paramers
    mnet.load_params(loaded_model, ctx=ctx, allow_missing=True)
    gammas = {k: v.data() for k, v in mnet.collect_params().items() if 'gamma' in k}
    return gammas


def load_gamma(gammas, dropout=False, ratio=0.2):
    # load gamma in the net to update mask
    loss = []
    for key, gamma in gammas.items():
        global_param.netMask[key] = assign_mask(gamma, global_param.netMask[key], key)
        mask = 1 - global_param.netMask[key]
        # if dropout, some fliters (ratio) will freeze
        if dropout:
            Dmask = global_dropout.get_select(mask, key, ratio)
            mask = (mask + Dmask) % 2

        loss.append(loss_gamma(mask, gamma))
    out = reduce(lambda x, y: x + y, loss)
    return out


def loss_gamma(mask, weight):
    # set L1 regularization for masked gamma
    # gamma only have one dimension
    channel = weight.shape[0]
    compress_target = nd.abs(mask * weight)
    all = nd.sum(compress_target)
    if not all.asscalar():
        return all
    w = compress_target / all
    loss = nd.sum(w * compress_target) / channel
    return loss


def get_sparse_ratio():
    sparse_dict = {}
    for k, v in global_param.netMask.items():
        all = reduce(lambda x, y: x * y, v.shape)
        pos = nd.sum(v).asscalar()
        sparse = 1.0 * pos / all
        sparse_dict[k] = sparse
    return sparse_dict
