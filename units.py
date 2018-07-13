from layers.params import global_param
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
    gammas = {}
    if not os.path.exists(loaded_model):
        for k, v in mnet.collect_params().items():
            if 'bias' in k:
                v.initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
                continue
            v.initialize(mxnet.initializer.Xavier(magnitude=3), ctx=ctx)
    else:
        mnet.load_params(loaded_model, ctx=ctx, allow_missing=True)
        for k, v in mnet.collect_params().items():
            if 'batchnorm' in k:
                if ('gamma' in k or 'var' in k) and sum(v.shape) == 0:
                    v.initialize(mxnet.initializer.Constant(1.0), ctx=ctx)
                elif ('beta' in k or 'mean' in k) and sum(v.shape) == 0:
                    v.initialize(mxnet.initializer.Constant(0.0), ctx=ctx)
        gammas = {k: v.data() for k, v in mnet.collect_params().items() if 'gamma' in k}
    return gammas


def get_sparse_ratio():
    sparse_dict = {}
    for k, v in global_param.netMask.items():
        all = reduce(lambda x, y: x * y, v.shape)
        pos = nd.sum(v).asscalar()
        sparse = 1.0 * pos / all
        sparse_dict[k] = sparse
    return sparse_dict
