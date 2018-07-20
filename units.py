from layers.params import global_param, global_dropout, sw
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
    gammas = [k for k in mnet.collect_params().keys() if 'gamma' in k]
    return gammas


def static(data):
    mu = nd.sum(nd.abs(data))
    std = nd.sum(data * data)
    count = nd.sum(data != 0)
    # all_count = reduce(lambda x, y: x * y, masked.shape)
    mu = mu / count
    std = std - count * nd.power(mu, 2)
    std = nd.sqrt(std / count)
    return mu, std


def load_gamma_test(mnet):
    # load gamma in the net to update mask
    loss_g = mxnet.gluon.loss.L1Loss()
    loss = []
    for key, value in mnet.collect_params().items():
        if not 'gamma' in key:
            continue
        gamma = value.data()
        mu = nd.mean(gamma)
        # global_param.netMask[key] = assign_mask(gamma, global_param.netMask[key], key)
        # mask = 1 - global_param.netMask[key]
        # # if dropout, some fliters (ratio) will freeze
        # if dropout:
        #     # Dmask = global_dropout.get_select(mask, key, ratio)
        #     k = int(ratio * gamma.shape[0])
        #     Dmask = nd.topk(gamma, k=k, ret_typ='mask', is_ascend=True)
        #     mask = (mask + Dmask) % 2
        tag_key = '_'.join(key.split('_')[1:])
        sw.add_scalar(tag=tag_key, value=mu.asscalar(), global_step=global_param.iter)
        target = nd.zeros_like(gamma).as_in_context(gamma.context)
        this_loss = loss_g(gamma / mu, target)
        loss.append(nd.sum(this_loss / gamma.shape[0]))

    out = reduce(lambda x, y: x + y, loss)
    return out / len(loss)


def load_gamma(mnet, dropout=False, ratio=0.2):
    # load gamma in the net to update mask
    loss_g = mxnet.gluon.loss.L1Loss()
    loss = []
    for key, value in mnet.collect_params().items():
        if not 'gamma' in key:
            continue
        gamma = value.data()
        global_param.netMask[key] = assign_mask(gamma, global_param.netMask[key], key)
        mask = 1 - global_param.netMask[key]
        # if dropout, some fliters (ratio) will freeze
        if dropout:
            # Dmask = global_dropout.get_select(mask, key, ratio)
            k = int(ratio * gamma.shape[0])
            Dmask = nd.topk(gamma, k=k, ret_typ='mask', is_ascend=True)
            mask = (mask + Dmask) % 2
        mine = mask * gamma
        target = nd.zeros_like(mine).as_in_context(mine.context)
        this_loss = loss_g(mine, target)
        loss.append(nd.sum(this_loss / gamma.shape[0]))

    out = reduce(lambda x, y: x + y, loss)
    return out / len(loss)


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


if __name__ == "__main__":
    from sphere_net import SphereNet20

    loaded_model = "log_bn_dy2/spherenet_bn_4dy.model"
    mnet = SphereNet20(use_bn=True)
    gammas = init_sphere(mnet, loaded_model)
