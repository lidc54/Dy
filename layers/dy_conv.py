from mxnet.gluon import nn
from mxnet import nd, initializer, autograd
from mxnet.gluon import loss, Trainer
import mxnet as mx
import math, random
from params import *


class origin_conv(nn.Conv2D):
    def __init__(self, channels, kernel_size, **kwargs):
        super(origin_conv, self).__init__(channels, kernel_size, **kwargs)


class new_conv(nn.Conv2D):
    def __init__(self, channels, kernel_size, **kwargs):
        # if isinstance(kernel_size, base.numeric_types):
        #     kernel_size = (kernel_size,)*2
        # assert len(kernel_size) == 2, "kernel_size must be a number or a list of 2 ints"

        super(new_conv, self).__init__(channels, kernel_size, **kwargs)

    # def  forward(self, x, *args):
    #     self.ctx = x.context
    #     self.set_params()
    #     return super(new_conv, self).forward(x, *args)
    def hybrid_forward(self, F, x, weight, bias=None):
        keys = self.params.keys()
        # self.assign_mask(keys)
        key = [key for key in keys if 'weight' in key][0]

        wmask = global_param.netMask[key] = \
            assign_mask(weight, global_param.netMask[key], key)
        # else:
        #     bmask = global_param.netMask[key] = \
        #         assign_mask(bias, global_param.netMask[key])
        # bmask
        bmask = wmask.copy()
        for i in range(2): bmask = nd.sum(bmask, axis=-1)
        # if global_param.iter % 1000 == 0:
        #     tag_key = '_'.join(key.split('_')[1:]) + '_KerLossNums'
        #     gls.sw.add_histogram(tag=tag_key, values=bmask.reshape(-1).copy().sort(), global_step=global_param.iter)
        bmask = nd.sum(bmask, axis=-1) > 0

        return super(new_conv, self).hybrid_forward(F, x, weight * wmask, bias * bmask)


# forward with mask
def assign_mask(weight, mask, key=None):
    # Calculate the mean and standard deviation of learnable parameters
    # maskshape = weight.shape
    # nd.sigmoid(weight),
    iter_ = global_param.iter
    masked = weight * mask
    mu = nd.sum(nd.abs(masked))
    std = nd.sum(masked * weight)
    count = nd.sum(masked != 0)
    all_count = reduce(lambda x, y: x * y, masked.shape)
    mu = mu / count
    std = std - count * nd.power(mu, 2)
    std = nd.sqrt(std / count)
    if key:
        global_param.netMask[key + '_muX'] = mu
        global_param.netMask[key + '_stdX'] = std
        mu = mu.asscalar()
        std = std.asscalar()
        tag_key = '_'.join(key.split('_')[1:])
        tag_shape = reduce(lambda x, y: x + 'X' + y, map(str, masked.shape))
        tag = [tag_key, tag_shape, str(all_count)]
        tag = '_'.join(tag)
        value = 1.0 * count.asscalar() / all_count
        gls.sw.add_scalar(tag=tag, value=value, global_step=global_param.iter)
        gls.sw.add_scalar(tag=tag + '_std', value=std, global_step=global_param.iter)
        gls.sw.add_scalar(tag=tag + '_mu', value=mu, global_step=global_param.iter)
    # Calculate the weight mask and bias mask with probability
    r = random.random()
    if math.pow(1 + gamma * iter_, -power) > r and iter_ < iter_stop:
        cond1 = (mask == 1) * (nd.abs(weight) < (0.9 * max(mu + c_rate * std, 0)))
        cond2 = (mask == 0) * (nd.abs(weight) > (1.1 * max(mu + c_rate * std, 0)))
        mask = mask - cond1 + cond2
    return mask


def constrain_kernal_num(mnet, ctx=mx.cpu()):
    # L1_loss = loss.L1Loss()
    exclude = ['alpha', 'bias', 'dense', '_muX', '_stdX']  # 'conv0_', 'conv1_', 'conv2_',

    def excluded(k):
        for i in exclude:
            if i in k:
                return True
        return False

    r = global_param.get_ratio()
    if r > 0:
        num_kernel = []
        for k, v in global_param.netMask.items():
            if excluded(k):
                continue
            w = mnet.collect_params()[k].data()
            # masked = w * v
            num_kernel.append(new_constrain_conv(w, k))

        # num_kernel = [constrain_conv(v, k) for k, v in global_param.netMask.items() if Not_excluded(k)]
        loss_nums = reduce(lambda x, y: x + y, num_kernel) / len(num_kernel)
        # Calculate the weight for mask
        # iter_ = global_param.iter
        # r = math.pow(1 + advanced_iter * gamma * iter_, -power)
        return loss_nums * r * nums_power
    return nd.zeros(1).as_in_context(ctx)


# using top_k
def new_constrain_conv(weight, name):
    out = nd.abs(weight.reshape(list(weight.shape[:2]) + [-1]))
    channel = out.shape[1]
    mu, std = global_param.netMask[name + '_muX'], global_param.netMask[name + '_stdX']
    tops = nd.topk(out, axis=-1, k=kept_in_kernel, ret_typ='mask')
    # let these in tops stay in activ & others be lost
    zeros = nd.zeros_like(out)
    keep = (1.10 * (mu + c_rate * std) - out) * tops
    keep = nd.where(keep > 0, keep, zeros)
    discard = (out - 0.9 * (mu + c_rate * std)) * (1 - tops)
    discard = nd.where(discard > 0, discard, zeros)
    # channels
    # keep_channels = nd.sum(keep != 0, axis=-1)
    # ones = nd.ones_like(keep_channels)
    # keep_channels = nd.where(keep_channels > 1, keep_channels, ones)
    # discard_channels = nd.sum(discard != 0, axis=-1)
    # discard_channels = nd.where(discard_channels > 1, discard_channels, ones)
    # nd.sum(keep, axis=-1) / keep_channels + nd.sum(discard, axis=-1) / discard_channels
    # channel = nd.sum(keep != 0, axis=-1) + nd.sum(discard != 0, axis=-1)
    # loss = nd.sum(keep) + nd.sum(discard)
    # keep_all = (1.10 * (mu + c_rate * std) - out)
    # keep_all = nd.where(keep_all > 0, keep_all, zeros)
    # for i in range(2): keep_all = nd.sum(keep_all, axis=-1)
    # discard_all = (out - 0.9 * (mu + c_rate * std))
    # discard_all = nd.where(discard_all > 0, discard_all, zeros)
    # for i in range(2): discard_all = nd.sum(discard_all, axis=-1)
    #
    loss_common = keep + discard
    # for i in range(2): loss_common = nd.sum(loss_common, axis=-1)
    # kept_ratio = global_param.get_kept_ratio()
    # all_common = nd.where((kept_ratio * loss_common) <= keep_all, loss_common, keep_all)
    # ac_none = nd.where(all_common <= discard_all, all_common, discard_all)
    loss = nd.sum(loss_common)  # nd.sum(ac_none)
    tag_key = '_'.join(name.split('_')[1:]) + '_KerLoss'
    gls.sw.add_scalar(tag=tag_key, value=loss.asscalar(), global_step=global_param.iter)

    # if global_param.iter > 150000: loss = loss * 2
    # if global_param.iter > 250000: loss = loss * 4
    # input*output;but output channels are almost same
    return loss  # / channel


# a convlution with constrain of limited paramers
def constrain_conv(weight, v, name):
    out = nd.abs(weight.reshape(list(weight.shape[:2]) + [-1]))
    mask = v.reshape(list(weight.shape[:2]) + [-1])
    mu, std = global_param.netMask[name + '_muX'], global_param.netMask[name + '_stdX']
    size = list(out.shape)
    # factor suppose as a shoulder
    factor = nd.sort(out, axis=-1)[:, :, -3].reshape(size[:2] + [1]) - mu.asscalar()
    bound = nd.ones_like(factor) * 0.5  # gamma has a great std now
    factor = nd.where(factor > bound, factor, bound)
    out = (out - mu) * zoom / factor
    out = nd.sum(mask * nd.sigmoid(out), axis=-1)

    channel = out.shape[1]
    out1 = nd.abs(out - kept_in_kernel)  # close to kept_in_kernel,for example 3;
    out2 = nd.abs(out)  # close t0 0
    out_c = nd.where(out1 < out2, out1, out2)
    # w = out_c / nd.sum(out_c)  # distribution the derivative
    constrain = nd.sum(out_c) / channel
    tag_key = '_'.join(name.split('_')[1:]) + '_K'
    gls.sw.add_scalar(tag=tag_key, value=constrain.asscalar(), global_step=global_param.iter)
    return constrain


# gamma in BN will be used for dynamic compress in a structure way.
# the method is discarded.we will compress relative items where mask masked.
class new_BN(nn.BatchNorm):
    def __init__(self):
        super(new_BN, self).__init__()

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        key = self.name + '_gamma'
        mask = global_param.netMask[key] = \
            assign_mask(gamma, global_param.netMask[key], key)
        return super(new_BN, self).hybrid_forward(F, x, gamma * mask, beta, running_mean, running_var)


if __name__ == "__main__":
    # a = nd.random.uniform(0, 1, shape=(1, 1, 5, 5))
    # lr = 0.1
    # net = nn.Sequential()
    #
    # cov = new_conv(2, 3, in_channels=1, padding=1, strides=1)
    # with net.name_scope():
    #     net.add(new_conv(2, 3, in_channels=1, padding=1, strides=1))
    #     net.add(new_conv(1, 3, in_channels=2, strides=1, padding=1))
    #
    # ctx = mx.cpu()
    # L = loss.L1Loss()
    # net.collect_params().initialize(init=initializer.Xavier(magnitude=3), ctx=ctx)
    # trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.0})
    # key = net.collect_params().keys()
    # a = a.as_in_context(ctx)
    #
    # global_param.set_param(key, ctx=ctx)
    #
    # for i in range(15000):
    #     with autograd.record():
    #         global_param.iter = i
    #         out = net[0](a)
    #         out = net[1](out)
    #         loss1 = L(out, a)
    #         print(loss1.asscalar())
    #     loss1.backward()
    #     gls.sw.add_scalar(tag='loss', value=loss1.asscalar(), global_step=i)
    #     trainer.step(1)
    #     if i != 0 and i % 1000 == 0:
    #         lr *= 0.5
    #         trainer.set_learning_rate(lr)
    # print('ok')
    a = new_conv(50, kernel_size=3, padding=(1, 1))
    print 'ok'
